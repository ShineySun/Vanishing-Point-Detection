import json
import numpy as np

# Tusimple Dataset Management Tool
class Tusimple_Manager(object):
    # initialize
    def __init__(self, root_dir = '/home/sun/tusimple'):
        # train data root path
        self.train_path = root_dir + '/train_set/'
        # test data root path
        self.test_path = root_dir + '/test_set/'
        # train data annotation files
        self.train_label_files = ['label_data_0313.json' , 'label_data_0531.json' , 'label_data_0601.json']
        # test data annotation files
        self.test_label_files = ['test_label.json']

        # train annotation data
        self.train_data = []
        # train data size
        self.train_size = 0
        # test annotation data
        self.test_data = []
        # test data size
        self.test_size = 0

        # number of train_instance 1
        self.train_instance_1 = []
        # number of train_instance 2
        self.train_instance_2 = []
        # number of train_instance 3
        self.train_instance_3 = []
        # number of train_instance 4
        self.train_instance_4 = []
        # number of train_instance 5
        self.train_instance_5 = []
        # number of train_instance 6
        self.train_instance_6 = []


        # number of test_instance 1
        self.test_instance_1 = []
        # number of test_instance 2
        self.test_instance_2 = []
        # number of test_instance 3
        self.test_instance_3 = []
        # number of test_instance 4
        self.test_instance_4 = []
        # number of test_instance 5
        self.test_instance_5 = []
        # number of test_instance 6
        self.test_instance_6 = []


    # load the data
    def tusimple_load_from_json(self):
        # train data load
        print("* Train Data Load Start")
        for idx, label_file in enumerate(self.train_label_files):
            # print("* {} : {} Load Start".format(idx, label_file))

            with open(self.train_path + label_file) as f:
                for line in f.readlines():
                    json_line = json.loads(line)
                    self.train_data.append(json_line)

            # print("* {} : {} Load Finish".format(idx, label_file))
        self.train_size = len(self.train_data)

        print("* Train Data Load Finish")

        # test data load
        print("* Test Data Load Start")

        for idx, label_file in enumerate(self.test_label_files):
            # print("* {} : {} Load Start".format(idx, label_file))

            with open(self.test_path + label_file) as f:
                for line in f.readlines():
                    json_line = json.loads(line)
                    self.test_data.append(json_line)

        self.test_size = len(self.test_data)

        print("* Test Data Load Finish")

    # split according to the number of instances
    def tusimple_split_instance(self):
        print("**-----------------------------------------------**")
        print("* Train Data Split Start")
        # train data split
        for idx,instance in enumerate(self.train_data):
            if len(instance['lanes']) == 1:
                self.train_instance_1.append(instance)
            elif len(instance['lanes']) == 2:
                self.train_instance_2.append(instance)
            elif len(instance['lanes']) == 3:
                self.train_instance_3.append(instance)
            elif len(instance['lanes']) == 4:
                self.train_instance_4.append(instance)
            elif len(instance['lanes']) == 5:
                self.train_instance_5.append(instance)
            elif len(instance['lanes']) == 6:
                self.train_instance_6.append(instance)


        print("num_train_instance_1 : {}".format(len(self.train_instance_1)))
        print("num_train_instance_2 : {}".format(len(self.train_instance_2)))
        print("num_train_instance_3 : {}".format(len(self.train_instance_3)))
        print("num_train_instance_4 : {}".format(len(self.train_instance_4)))
        print("num_train_instance_5 : {}".format(len(self.train_instance_5)))
        print("num_train_instance_6 : {}".format(len(self.train_instance_6)))


        print("* Train Data Split Finish")
        print("**-----------------------------------------------**")

        print("* Test Data Split Start")

        # test data split
        for idx,instance in enumerate(self.test_data):
            if len(instance['lanes']) == 1:
                self.test_instance_1.append(instance)
            elif len(instance['lanes']) == 2:
                self.test_instance_2.append(instance)
            elif len(instance['lanes']) == 3:
                self.test_instance_3.append(instance)
            elif len(instance['lanes']) == 4:
                self.test_instance_4.append(instance)
            elif len(instance['lanes']) == 5:
                self.test_instance_5.append(instance)
            elif len(instance['lanes']) == 6:
                self.test_instance_6.append(instance)

        print("num_test_instance_1 : {}".format(len(self.test_instance_1)))
        print("num_test_instance_2 : {}".format(len(self.test_instance_2)))
        print("num_test_instance_3 : {}".format(len(self.test_instance_3)))
        print("num_test_instance_4 : {}".format(len(self.test_instance_4)))
        print("num_test_instance_5 : {}".format(len(self.test_instance_5)))
        print("num_test_instance_6 : {}".format(len(self.test_instance_6)))

        print("* Test Data Split Finish")

        print("**-----------------------------------------------**")

    def get_instance(self, option = 0):
        if option == 0:
            print("ALL DATASET IS RETURNED")
            return self.train_data, self.test_data
        elif option == 1:
            print("DATASET 1 IS RETURNED")
            return self.train_instance_1, self.test_instance_1
        elif option == 2:
            print("DATASET 2 IS RETURNED")
            return self.train_instance_2, self.test_instance_2
        elif option == 3:
            print("DATASET 3 IS RETURNED")
            return self.train_instance_3, self.test_instance_3
        elif option == 4:
            print("DATASET 4 IS RETURNED")
            return self.train_instance_4, self.test_instance_4
        elif option == 5:
            print("DATASET 5 IS RETURNED")
            return self.train_instance_5, self.test_instance_5
        elif option == 6:
            print("DATASET 6 IS RETURNED")
            return self.train_instance_6, self.test_instance_6
        else:
            print("OPTION(0, 1, 2, 3, 4, 5, 6) IS REQUIRED")
            return None, None


    def get_lanes(self, raw_data):
        # Define return list
        lanes = []
        # extract lanes
        for lane_idx, lane_x_points in enumerate(raw_data['lanes'], 0):
            vertices = list(filter(lambda xy_pair : xy_pair[0] > 0, zip(lane_x_points, raw_data['h_samples'])))

            if len(vertices) == 0: continue

            vertices = np.array(vertices, dtype=np.float)

            lanes.append(np.flip(vertices, axis=0))

        return lanes

    def split_train_test(self, datas, test_size=5, seq_len=5):
        # train_sets & test_sets
        train_sets = []
        test_sets = []

        for data in datas:
            # normalize
            data[:, 0] /= 1280.0
            data[:, 1] /= 720.0

            train_size = len(data) - test_size

            train_set = data[:train_size]
            test_set = data[train_size-seq_len:]

            train_sets.append(train_set)
            test_sets.append(test_set)

        return train_sets, test_sets

    def make_sequence_data(self, data, seq_len = 5):
        l = len(data)

        data_x = []
        data_y = []

        for i in range(l-seq_len):
            # y points -> x value
            train_seq = data[i:i+seq_len, 1]
            # x points -> y value
            test_label = data[i+seq_len:i+seq_len+1, 0]

            data_x.append(train_seq)
            data_y.append(test_label)

        return np.array(data_x), np.array(data_y)

    def make_test_sequence_data(self, data, seq_len):
        l = len(data)

        data_x = []

        for i in range(l-seq_len):
            train_seq = data[i:i+seq_len]

            data_x.append(train_seq)

        return np.array(data_x, dtype=np.float)









if __name__ == '__main__':
    # create
    tusimple_manager = Tusimple_Manager()

    # load data
    tusimple_manager.tusimple_load_from_json()


    # split data
    tusimple_manager.tusimple_split_instance()
