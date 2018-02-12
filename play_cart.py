import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        # for t in range(200):
        while True:
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            # action = env.action_space.sample()
            action = random.randrange(0, 2)
            print(action)


            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break

def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = random.randrange(0,2)
            # do it!
            observation, reward, done, info = env.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                action = [1 - data[1], data[1]]

                # saving our training data
                training_data.append([data[0], action])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)

    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 2, activation='relu')
    network = dropout(network, 0.8)

    # network = fully_connected(network, 128, activation='relu')
    # network = dropout(network, 0.8)
    #
    # network = fully_connected(network, 256, activation='relu')
    # network = dropout(network, 0.8)
    #
    # network = fully_connected(network, 512, activation='relu')
    # network = dropout(network, 0.8)
    #
    # network = fully_connected(network, 256, activation='relu')
    # network = dropout(network, 0.8)
    #
    # network = fully_connected(network, 128, activation='relu')
    # network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network)

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model



def play():
    training_data = initial_population()
    model = train_model(training_data)

    scores = []
    choices = []
    max_score = 0
    min_score = 0
    steps = 1000
    for _ in range(5):
        game_memory = []
        last_score = 0
        this_min_score = 0
        for each_game in range(100):
            print('GAME: #' + str(each_game))
            score = 0
            prev_obs = []
            env.reset()
            for _ in range(steps):
            # while True:
                # env.render()

                if len(prev_obs)==0:
                    action = random.randrange(0,2)
                else:
                    action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
                choices.append(action)

                new_observation, reward, done, info = env.step(action)

                action = [1 - action, action]
                if len(prev_obs) > 0:
                    game_memory.append([prev_obs, action])

                prev_obs = new_observation
                score+=reward
                if new_observation[0] < -2.4 or new_observation[0] > 2.4:
                    break;
                # if done:
                #     print('Score:', score)
                #     break
            print('Score:', score)
            scores.append(score)

            if score > max_score:
                max_score = score
            if score < this_min_score:
                this_min_score = score
            last_score = score
        if len(game_memory) > 0 and this_min_score > min_score:
            model = train_model(game_memory, model)
            min_score = this_min_score
    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    print('Max Score:', max_score)
    print(score_requirement)

# some_random_games_first()
play()
