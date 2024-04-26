import imageio
images = []
for i in range(0,360):
    images.append(imageio.imread('final_roll_vid/'+str(i)+'.png'))
imageio.mimsave('final_roll_vid/roll_vid.gif', images)
