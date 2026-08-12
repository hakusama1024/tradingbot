# TradingBot 策略总览

这套系统不是单一策略，而是多个互相隔离的 profile。每个 profile 都有自己的定位、运行频率、runtime 数据和风控边界，方便做 A/B test，也方便单独关闭某一套策略。

## 当前策略地图

| 分类 | Profile | 作用 | 交易风格 | 运行频率 |
| --- | --- | --- | --- | --- |
| 核心轮动 | `paper` | 主 paper 账户核心仓 | SPY/QQQ/SMH 战术轮动 | launchd 每 15 分钟检查；策略约 15 个交易日再平衡 |
| 个股机会扫描 | `paper_stockscan` | 主 paper 账户的卫星策略 | Minervini / leader-continuation 个股突破 | 盘中每 10 分钟扫描 |
| CAN SLIM 实验 | `paper_canslim` | A/B test 账户 | CAN SLIM 选股 + Minervini 执行 | 盘中每 10 分钟扫描 |
| TriCore 测试 | `paper_spy_alpha` | 独立 ETF 轮动测试 | SPY/QQQ/SMH adaptive momentum | launchd 每 15 分钟检查；策略约 15 个交易日再平衡 |
| 卖 Put 收益研究 | `paper_put_income` | 期权收入模拟器 | Cash-secured put 扫描和本地 paper 模拟 | 手动运行或单独排程 |
| 实盘 | `live` | 真钱账户 profile | 当前默认关闭 | 需要明确启用才安装服务 |

## 1. 核心轮动：`paper`

`paper` 是主 paper 账户的核心策略，当前使用 SPY-alpha / TriCore tactical rotation。它不追求每天交易，而是在 `SPY`、`QQQ`、`SMH` 之间做低频轮动。

核心逻辑：

- 用 161 个交易日动量给 `SPY`、`QQQ`、`SMH` 排名。
- 用 200 日均线判断市场和资产趋势。
- 如果第一名明显领先，集中持有第一名。
- 如果前两名差距不大，持有前两名。
- 用 20 日波动率做仓位缩放，目标约 17% 年化波动。
- 慢速再平衡，减少无意义换手。

为什么需要它：

- 它是账户的核心 beta / sector rotation 暴露。
- 它比频繁个股交易更稳。
- 它能参与强趋势，同时尽量控制回撤。

主要缺点：

- 它会错过一些快速个股突破。
- 它不是高频 stock picker。

## 2. 个股机会扫描：`paper_stockscan`

`paper_stockscan` 是后来加回来的卫星策略。它的目的不是替代 `paper` 的 TriCore 核心仓，而是恢复“盘中每 10 分钟看一遍市场有没有个股机会”的能力。

核心逻辑：

- 扫描广泛的流动性美股 universe。
- 用 Minervini trend template / base / pivot 过滤。
- 寻找强势股的 leader continuation 或 breakout setup。
- 满足规则时提交带保护止损的买入单。
- 盘中 10:00 到 15:59 ET，每 10 分钟运行一次。

为什么需要它：

- TriCore 低频轮动会错过个股机会。
- `paper_stockscan` 专门负责更快发现个股突破。
- 它和核心仓隔离，便于看清到底是谁贡献了收益。

重要边界：

- `OVERLAY_ENABLED=0`，避免它重复买 `SMH`。
- 已有持仓会继续管理止损和盈利保护。
- 新开仓仍然受市场状态、形态质量、突破状态和风控约束。

## 3. CAN SLIM 实验：`paper_canslim`

`paper_canslim` 是 A/B test 账户，组合方式是：

- CAN SLIM 负责选“谁值得看”。
- Minervini 负责判断“什么时候买、怎么买、怎么卖”。

CAN SLIM 更重视：

- 当前季度 EPS / revenue 增长。
- 年度增长质量。
- 新高和强势股。
- 行业或市场领导地位。
- 类似机构资金推动的趋势。

当前结论：

- 这套策略最近表现不如主 `paper`。
- 它仍然有价值，因为可以作为 candidate discovery。
- 但在实盘前，不能把它当成主策略。

## 4. TriCore Momentum 测试：`paper_spy_alpha`

`paper_spy_alpha` 是独立的 ETF 轮动测试账户。它和主 `paper` 的 TriCore 逻辑属于同一类，但独立运行，方便做对照。

核心逻辑：

- 在 `SPY`、`QQQ`、`SMH` 之间做 adaptive momentum。
- 用 200 日均线做趋势过滤。
- 用波动率控制仓位。
- 慢速再平衡，不追盘中噪音。

为什么保留它：

- 用来评估 ETF 轮动本身是否有效。
- 和个股扫描、CAN SLIM 分开，避免结果混在一起。
- 可以作为“简单 ETF alpha”的对照组。

## 5. 卖 Put 收益研究：`paper_put_income`

`paper_put_income` 是 cash-secured put 的本地 paper 模拟器。它目前不会真的通过 Alpaca 下期权单，只用于扫描、模拟和规则验证。

核心逻辑：

- 只做 cash-secured put，不裸卖。
- 标的优先选择流动性 ETF 和质量较好的大盘股。
- DTE 主要看 30 到 60 天，默认接近 45 DTE。
- Delta 主要控制在 0.16 到 0.30。
- 标的必须在 200 日均线上方。
- 要求最低 IV 和期权流动性。
- 盈利 50% 时买回。
- 21 DTE 左右开始管理。
- 限制单标的担保金额和组合总担保金额。

为什么需要它：

- 它更像“收租型”低波动策略。
- 它不是为了在强牛市里 beat `SPY`、`QQQ` 或 `SMH`。
- 如果以后接入真实 options execution，可以作为单独的 income sleeve。

重要限制：

- 当前 10 年回测使用的是基于正股日线和 realized volatility 的合成期权价格。
- 这个回测可以验证规则方向，但不等于真实历史期权报价回测。
- 真正实盘前，需要接入更可靠的 options quote / fill / assignment 处理。

## 6. 通知和新闻

系统支持 ntfy 推送：

- 下单提醒。
- 早盘 scan 总结。
- 每日总结。
- 每周总结。
- 社交/RSS 监控。
- Market news radar。

当前 `live` profile 默认关闭。公开 repo 里不应该出现真实 ntfy topic、API key、broker key、账户 ID 或数据库。

## 推荐运行方式

- `paper` 负责低频核心 ETF 轮动。
- `paper_stockscan` 负责 10 分钟个股机会扫描。
- `paper_canslim` 保持实验账户，不直接作为主策略。
- `paper_spy_alpha` 保持独立 ETF 轮动对照。
- `paper_put_income` 继续 paper-only，直到 options execution 验证完成。
- `live` 默认关闭，只有在明确承担真金白银风险时才启用。

## 安全规则

- 不提交 `.env`。
- 不提交 Alpaca key、OpenAI key、ntfy topic、账户 ID 或数据库。
- 所有交易信号都应视为实验结果。
- Paper performance 不等于 live performance。
- 修改策略前优先新增独立 profile，不要直接污染已有策略。

