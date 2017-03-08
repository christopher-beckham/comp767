import gym
import time


"""
0 = stay
1 = stay
2 = up
3 = down
4 = up
5 = down
"""
def test_game():
    env = gym.make("Pong-v0")
    observation = env.reset()
    for i in range(1000):
        env.render()
        #action = env.action_space.sample() # your agent here (this takes random actions)
        action = 5
        
        observation, reward, done, info = env.step(action)
        #print "i, obs, reward, done, info", observation.shape, reward, done, info
        print i, action
        time.sleep(0.3)


if __name__ == '__main__':
    test_game()

    #env = gym.make("Pong-v0")
    # what is the action space?

