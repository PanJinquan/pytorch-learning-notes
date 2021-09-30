# ***************** gen_data.py *******************
import time
import tensorwatch as tw
import random

'''
《tensorwatch——可视化深度学习库的使》https://blog.csdn.net/qq_29592829/article/details/90517303
'''


# create watcher, notice that we are not logging anything
def tensorwatch1():
    w = tw.Watcher()
    for step in range(10000):
        loss = random.random() * 10
        train_accuracy = random.random()

        # we are just observing variables
        # observation has no cost, nothing gets logged anywhere
        w.observe(step=step, loss=loss, train_accuracy=train_accuracy)

        time.sleep(1)
        print("step:{},loss:{}".format(step, loss))


def tensorwatch2():
    w = tw.Watcher()
    for step in range(10000):
        iteration = step
        face_loss = (step%20)/20
        head_loss = (step%10)/10
        loss = face_loss + head_loss
        w.observe(step=iteration, loss=loss, face_loss=face_loss, body_loss=head_loss)

        time.sleep(1)
        print("step:{},loss:{},face_loss:{}, head_loss:{}".format(iteration, loss, face_loss, head_loss))


if __name__ == "__main__":
    tensorwatch2()
