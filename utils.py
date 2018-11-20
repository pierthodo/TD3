import numpy as np

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
        def __init__(self):
                self.storage = []
                self.new_episode_index = []
                self.current_episode = 0

        # Expects tuples of (state, next_state, action, reward, done)
        def add(self, data, episode_num=None):
                self.storage.append(data)

                if episode_num != self.current_episode:
                    self.current_episode += 1
                    assert episode_num == self.current_episode

        def sample(self, batch_size=100,N_backprop=1):
                # sample random indices from buffer
                ind = np.random.randint(0, len(self.storage), size=batch_size)                                
                x, y, u, r, d = [], [], [], [], []

                current_size = 0
                current_i    = 0
                while current_size < batch_size:
                        i = ind[current_i]
                        
                        # make sure the sampled indices are not over two episodes
                        if sum([i - j in self.new_episode_index for j in range(N_backprop)]) == 0:
                            current_size += 1
                            curr_x, curr_y, curr_u, curr_r, curr_d = [], [], [], [], []
                            
                            for j in range(N_backprop):
                                idx = i - (N_backprop - j + 1)
                                if idx < 0:
				    idx =0
                                
                                X, Y, U, R, D = self.storage[idx]

                                curr_x.append(np.array(X, copy=False))
                                curr_y.append(np.array(Y, copy=False))
                                curr_u.append(np.array(U, copy=False))
                                curr_r.append(np.array(R, copy=False))
                                curr_d.append(np.array(D, copy=False))
                                
                            x.append(np.array(curr_x, copy=False))
                            y.append(np.array(curr_y, copy=False))
                            u.append(np.array(curr_u, copy=False))
                            r.append(np.array(curr_r, copy=False))
                            d.append(np.array(curr_d, copy=False))

                        else: 
                            pass
                            # print('invalid index {}'.format(i))
                        current_i += 1

                out = np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, N_backprop, 1), np.array(d).reshape(-1, N_backprop, 1)
                # out = [item.squeeze(axis=1) for item in out]
                return out
