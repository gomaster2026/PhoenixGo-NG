/*
 * Tencent is pleased to support the open source community by making PhoenixGo available.
 * 
 * Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
 * 
 * Licensed under the BSD 3-Clause License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     https://opensource.org/licenses/BSD-3-Clause
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "mcts_engine.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

#include <glog/logging.h>

#include "common/str_utils.h"
#include "model/zero_model.h"
#include "model/trt_zero_model.h"
#include "dist/dist_zero_model_client.h"
#include "dist/async_dist_zero_model_client.h"

static thread_local std::random_device g_random_device;
static thread_local std::minstd_rand g_random_engine(g_random_device());

MCTSEngine::MCTSEngine(const MCTSConfig &config)
    : m_config(config),
      m_root(nullptr),
      m_board(!config.disable_positional_superko()),
      m_eval_task_queue(config.eval_task_queue_size()),
      m_model_global_step(0),
      m_is_searching(false),
      m_simulation_counter(0),
      m_num_moves(0),
      m_gen_passes(0),
      m_monitor(this),
      m_debugger(this)
{
    // setup eval threads
    if (m_config.model_config().enable_mkl()) {
        ZeroModel::SetMKLEnv(m_config.model_config());
    }
    std::vector<int> gpu_list;
    for (const std::string &gpu: SplitStr(m_config.gpu_list(), ',')) {
        gpu_list.push_back(gpu.empty() ? 0 : std::stoi(gpu));
    }
    for (int i = 0; i < m_config.num_eval_threads(); ++i) {
        std::unique_ptr<ZeroModelBase> model;
        if (m_config.enable_dist()) {
            const auto &addr = m_config.dist_svr_addrs(i % m_config.dist_svr_addrs_size());
            if (m_config.enable_async()) {
                model.reset(new AsyncDistZeroModelClient(SplitStr(addr, ','), m_config.dist_config()));
            } else {
                model.reset(new DistZeroModelClient(addr, m_config.dist_config()));
            }
        } else if (m_config.model_config().enable_tensorrt()) {
            model.reset(new TrtZeroModel(gpu_list[i % gpu_list.size()]));
        } else {
            model.reset(new ZeroModel(gpu_list[i % gpu_list.size()]));
        }
        m_eval_threads_init_wg.Add();
        m_eval_threads.emplace_back(&MCTSEngine::EvalRoutine, this, std::move(model));
    }

    // setup search threads
    for (int i = 0; i < m_config.num_search_threads(); ++i) {
        m_search_threads.emplace_back(&MCTSEngine::SearchRoutine, this);
    }

    // setup delete thread & tree root
    m_delete_thread = std::thread(&MCTSEngine::DeleteRoutine, this);
    ChangeRoot(nullptr);

    // wait
    LOG(INFO) << "MCTSEngine: waiting all eval threads init";
    m_eval_threads_init_wg.Wait();
    LOG(INFO) << "MCTSEngine: all eval threads init done";

    if (m_config.enable_background_search()) {
        SearchResume();
    }
}

MCTSEngine::~MCTSEngine()
{
    LOG(INFO) << "~MCTSEngine: Deconstructing MCTSEngine";
    m_search_threads_conductor.Terminate();
    LOG(INFO) << "~MCTSEngine: Waiting search threads terminate";
    for (auto &th: m_search_threads) {
        th.join();
    }
    LOG(INFO) << "~MCTSEngine: Waiting eval threads terminate";
    m_eval_task_queue.Close();
    for (auto &th: m_eval_threads) {
        th.join();
    }
    LOG(INFO) << "~MCTSEngine: Waiting delete thread terminate";
    m_delete_queue.Push(m_root);
    m_delete_queue.Close();
    m_delete_thread.join();
    LOG(INFO) << "~MCTSEngine: Deconstruct MCTSEngin succ";
}

void MCTSEngine::Reset(const std::string &init_moves)
{
    SearchPause();
    ChangeRoot(nullptr);
    m_board.CopyFrom(GoState(!m_config.disable_positional_superko()));
    m_simulation_counter = 0;
    m_num_moves = (init_moves.size() + 1) / 3;
    m_moves_str = init_moves;
    m_gen_passes = 0;
    m_byo_yomi_timer.Reset();

    for (size_t i = 0; i < init_moves.size(); i += 3) {
        GoCoordId x, y;
        GoFunction::StrToCoord(init_moves.substr(i, 2), x, y);
        m_board.Move(x, y);
        m_root->move = GoFunction::CoordToId(x, y);
    }

    if (m_config.enable_background_search()) {
        SearchResume();
    }
}

void MCTSEngine::Move(GoCoordId x, GoCoordId y)
{
    if (!m_byo_yomi_timer.IsEnable()) {
        auto &c = m_config.time_control();
        if (c.main_time() > 0 || c.byo_yomi_time() > 0) {
            m_byo_yomi_timer.Set(c.main_time(), c.byo_yomi_time());
        }
    }

    SearchPause();

    if (GoFunction::IsResign(x, y)) {
        LOG(INFO) << "Move: resign";
        return;
    }

    int ret = m_board.Move(x, y);
    CHECK_EQ(ret, 0) << "Move: failed, " << GoFunction::CoordToStr(x, y) << ", ret" << ret;

    ++m_num_moves;
    if (m_moves_str.size()) m_moves_str += ",";
    m_moves_str += GoFunction::CoordToStr(x, y);
    LOG(INFO) << "Move: " << m_moves_str;

    ChangeRoot(FindChild(m_root, GoFunction::CoordToId(x, y)));
    m_root->move = GoFunction::CoordToId(x, y);

    m_debugger.UpdateLastMoveDebugStr();
    LOG(INFO) << m_debugger.GetLastMoveDebugStr();
    m_debugger.PrintTree(1, 10, GoFunction::CoordToStr(x, y) + ",");

    m_byo_yomi_timer.HandOff();

    if (m_pending_config) {
        m_config = *m_pending_config;
        m_pending_config = nullptr;
        LOG(INFO) << "reload config succ: \n" << m_config.DebugString();
    }

    if (!m_config.disable_double_pass_scoring() && m_board.IsDoublePass()) {
        LOG(INFO) << "Move: double pass, game ends";
        return;
    }

    if (m_config.enable_background_search()) {
        SearchResume();
    } else {
        m_simulation_counter = 0;
    }
}

void MCTSEngine::GenMove(GoCoordId &x, GoCoordId &y)
{
    std::vector<int> visit_count;
    float v_resign;
    GenMove(x, y, visit_count, v_resign);
}

void MCTSEngine::GenMove(GoCoordId &x, GoCoordId &y, std::vector<int> &visit_count, float &v_resign)
{
    if (!m_byo_yomi_timer.IsEnable()) {
        auto &c = m_config.time_control();
        if (c.main_time() > 0 || c.byo_yomi_time() > 0) {
            m_byo_yomi_timer.Set(c.main_time(), c.byo_yomi_time());
        }
    }

    if (m_config.enable_pass_pass() && m_board.GetLastMove() == GoComm::COORD_PASS && !IsPassDisable()) {
        x = y = GoComm::COORD_PASS;
        ++m_gen_passes;
        visit_count = GetVisitCount(m_root);
        v_resign = 1.0f;
        return;
    }

    Search();
    visit_count = GetVisitCount(m_root);

    int move = GoComm::COORD_UNSET;
    if (m_config.genmove_temperature() == 0.0f) {
        move = GetBestMove(v_resign);
    } else {
        move = GetSamplingMove(m_config.genmove_temperature());
        v_resign = 1.0f;
    }
    GoFunction::IdToCoord(move, x, y);

    if (move == GoComm::COORD_PASS) {
        ++m_gen_passes;
    }
}

bool MCTSEngine::Undo()
{
    if (m_num_moves == 0) {
        return false;
    }
    Reset(m_moves_str.substr(0, m_moves_str.size() - 3));
    return true;
}

const GoState &MCTSEngine::GetBoard()
{
    return m_board;
}

MCTSConfig &MCTSEngine::GetConfig()
{
    return m_config;
}

void MCTSEngine::SetPendingConfig(std::unique_ptr<MCTSConfig> config)
{
    m_pending_config = std::move(config);
}

MCTSDebugger &MCTSEngine::GetDebugger()
{
    return m_debugger;
}

int MCTSEngine::GetModelGlobalStep()
{
    return m_model_global_step;
}

ByoYomiTimer &MCTSEngine::GetByoYomiTimer()
{
    return m_byo_yomi_timer;
}

TreeNode *MCTSEngine::InitNode(TreeNode *node, TreeNode *fa, int move, float prior_prob)
{
    node->fa = fa;
    node->ch = nullptr;
    node->ch_len = 0;
    node->size = 1;
    node->expand_state = k_unexpanded;
    node->move = move;
    node->visit_count = 0;
    node->virtual_loss_count = 0;
    node->total_action = 0;
    node->prior_prob = prior_prob;
    node->value = NAN;
    return node;
}

TreeNode *MCTSEngine::FindChild(TreeNode *node, int move)
{
    TreeNode *ch = n