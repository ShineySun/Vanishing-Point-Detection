import json

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



if __name__ == '__main__':
    # create
    tusimple_manager = Tusimple_Manager()

    # load data
    tusimple_manager.tusimple_load_from_json()


    # split data
    tusimple_manager.tusimple_split_instance()
