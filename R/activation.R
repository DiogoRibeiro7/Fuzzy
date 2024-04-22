ActivationFunctions <- list(
  sigmoid = function(x) {
    1 / (1 + exp(-x))
  },
  elu = function(x, alpha = 1.0) {
    ifelse(x > 0, x, alpha * (exp(x) - 1))
  },
  relu = function(x) {
    pmax(0, x)
  },
  tanh = function(x) {
    tanh(x)
  },
  softmax = function(x) {
    exp_x <- exp(x - max(x))
    exp_x / sum(exp_x)
  },
  gelu = function(x) {
    0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
  },
  prelu = function(x, alpha = 0.01) {
    ifelse(x > 0, x, alpha * x)
  },
  leaky_relu = function(x, alpha = 0.01) {
    ifelse(x > 0, x, alpha * x)
  },
  parametric_relu = function(x, alpha) {
    ifelse(x > 0, x, alpha * x)
  },
  relu6 = function(x) {
    pmin(pmax(0, x), 6)
  },
  binary_step = function(x) {
    ifelse(x >= 0, 1, 0)
  },
  identity = function(x) {
    x
  },
  swish = function(x, beta = 1.0) {
    x * sigmoid(beta * x)
  },
  hard_swish = function(x) {
    x * pmin(1, pmax(0, (x + 3) / 6))
  },
  softplus = function(x) {
    log(1 + exp(x))
  },
  selu = function(x, alpha = 1.67326, scale = 1.0507) {
    scale * ifelse(x > 0, x, alpha * (exp(x) - 1))
  },
  mish = function(x) {
    x * tanh(log(1 + exp(x)))
  },
  rrelu = function(x, lower = 0.125, upper = 0.333333) {
    alpha <- runif(1, min = lower, max = upper)
    ifelse(x > 0, x, alpha * x)
  },
  softsign = function(x) {
    x / (1 + abs(x))
  },
  hard_tanh = function(x, min_val = -1.0, max_val = 1.0) {
    pmin(pmax(x, min_val), max_val)
  },
  hard_sigmoid = function(x) {
    pmin(pmax(0, 0.2 * x + 0.5), 1)
  },
  tanh_shrink = function(x) {
    x - tanh(x)
  },
  soft_shrink = function(x, lambda_ = 0.5) {
    ifelse(x > lambda_, x - lambda_, ifelse(x < -lambda_, x + lambda_, 0))
  }
)
