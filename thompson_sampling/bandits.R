library(vioplot)
library(nnet)

set.seed(2) # 2
k = 5
means = rnorm(k, 0, 1)
# generate a violin plot
n = 1000
vioplot(
  rnorm(n, means[1], 1),
  rnorm(n, means[2], 1),
  rnorm(n, means[3], 1),
  rnorm(n, means[4], 1),
  rnorm(n, means[5], 1),
  names=1:5,
  col="grey"
)
abline(h=0, lty="dashed")

# -----------

eps.greedy = function(means, eps, num.iters) {
  k = length(means)
  Q = rep(0, k)
  N = rep(0, k)
  avg.rewards = rep(0, num.iters)
  for(iter in 1:num.iters) {
    if(runif(1,0,1) < eps) {
      A = sample(1:k, 1)
    } else {
      A = which.is.max(Q)
    }
    R = rnorm(1, means[A], 1)
    N[A] = N[A] + 1
    Q[A] = Q[A] + ((1/N[A])*(R - Q[A]))
    # compute average reward
    avg.rewards[iter] = ((N / sum(N)) %*% Q)[1]
  }
  print(Q)
  print(N)
  return(avg.rewards)
}

thompson.gauss = function(means, num.iters) {
  # prior is an N(0,1)
  mean.prior = 0
  sd.prior = 1
  Q = rep(0, k)
  N = rep(0, k)
  avg.rewards = rep(0, num.iters)
  for(iter in 1:num.iters) {
    # for each arm i = 1...k, sample \theta_i from posterior distribution
    theta.samples = c()
    for(a in 1:k) {
      theta.samples = c(theta.samples, rnorm(1, Q[a], 1 / (N[a]+1) ))
    }
    # choose the arm that is the argmax of these samples
    A = which.max(theta.samples)
    R = rnorm(1, means[A], 1)
    N[A] = N[A] + 1
    Q[A] = Q[A] + ((1/N[A])*(R - Q[A]))
    # compute average reward
    avg.rewards[iter] = ((N / sum(N)) %*% Q)[1]
  }
  print(Q)
  print(N)
  return(avg.rewards)
}

ucb = function(means, c, num.iters) {
  k = length(means)
  Q = rep(0, k)
  N = rep(0, k)
  avg.rewards = rep(0, num.iters)
  for(iter in 1:num.iters) {
    confs = c()
    for(a in 1:k) {
      confs = c(confs, Q[a] + (c*sqrt( log(iter) / (N[a]+1) )) )
    }
    A = which.max(confs)
    R = rnorm(1, means[A], 1)
    N[A] = N[A] + 1
    Q[A] = Q[A] + ((1/N[A])*(R - Q[A]))
    # compute average reward
    avg.rewards[iter] = ((N / sum(N)) %*% Q)[1]
  }
  print(Q)
  print(N)
  return(avg.rewards)
}

# look at epsilon greedy
# eps=0 can 'fluke' and be the best often. i think this is due to the random tie breaking at the start picking
# the right arm(s) by chance. i'd expect this to happen less often with the more arms you have
set.seed(0)
plot(eps.greedy(means, 0.1, 1000),type="l")
lines(eps.greedy(means, 0.01, 1000),type="l",col="red")
lines(eps.greedy(means, 0, 1000),type="l", col="green")
lines(eps.greedy(means, 1, 1000),type="l", lty="dashed", col="black")
legend("bottomright",legend=c(0.1, 0.01, 0, 1),col=c("black","red","green","black"),lty=c("solid","solid","solid","dashed"),lwd=2)

# look at thompson sampling
set.seed(0)
plot(thompson.gauss(means, 1000), type="l", ylim=c(-1,1.5), xlab="time", ylab="average reward")
lines(eps.greedy(means, 0.1, 1000), type="l",col="blue")
lines(eps.greedy(means, 0.01, 1000), type="l",col="red")
lines(eps.greedy(means, 0, 1000), type="l",col="green")
lines(eps.greedy(means, 1, 1000), type="l", lty="dashed", col="black")
legend("bottomright",legend=c("thompson", "eps=0.1", "eps=0.01", "eps=1"),col=c("black","blue", "red","black"),lty=c("solid","solid","solid","dashed"),lwd=2)

# look at ucb sampling
set.seed(0)
plot(ucb(means, c=0, 1000),type="l", ylim=c(0,1.5))
lines(ucb(means, c=0.5, 1000),col="red")
lines(ucb(means, c=1, 1000),col="blue")
lines(ucb(means, c=2, 1000),col="green")
legend("bottomright",legend=c("c=0", "c=0.5", "c=1", "c=2"),col=c("black","red","blue","green"),lty="solid",lwd=2)

# look at all of them together
set.seed(0)
plot(eps.greedy(means, 0.1, 1000), type="l", ylim=c(0,1.5))
lines(ucb(means, 2, 1000),col="red")


