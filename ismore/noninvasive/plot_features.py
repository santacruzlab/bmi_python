plt.figure()
plt.plot(r_feats)
plt.show(block = False)
plt.plot(m_feats)
plt.show(block = False)

plt.figure()
# plot all the feature windows corresponding to rest
plt.plot((features[5][np.where(labels[5][:,0] == 0)[0],:].T), color = 'blue')
# plot all the feature windows corresponding to move
plt.plot((features[5][np.where(labels[5][:,0] == 1)[0],:].T), color = 'red')

plt.figure()
# plot mean of all the feature windows corresponding to rest
plt.plot(np.mean(features[5][np.where(labels[5][:,0] == 0)[0],:].T, axis = 1), color = 'blue')
# plot mean of all the feature windows corresponding to move
plt.plot(np.mean(features[5][np.where(labels[5][:,0] == 1)[0],:].T, axis = 1), color = 'red')
plt.show(block = False)


plt.figure()
plt.imshow(eeg_power, interpolation = 'none')
plt.plot([rest_end_idxs_eeg, rest_end_idxs_eeg],[4, 8], color = 'green')
plt.plot([mov_start_idxs_eeg, mov_start_idxs_eeg],[0, 30], color = 'red')
plt.plot([rest_start_idxs_eeg, rest_start_idxs_eeg],[4, 8], color = 'brown')
plt.plot([mov_end_idxs_eeg, mov_end_idxs_eeg],[4, 8], color = 'black')
plt.show(block = False)

