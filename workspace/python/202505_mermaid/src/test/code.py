class Stage1:
    name = "stage1"
    description = "abc"
    inputs = ["in1_1", "in1_2", "in1_3"]
    outputs = ["out1_1", "out1_2", "out1_3"]

    def run(self):
        print("Do something")


class Stage2:
    name = "stage2"
    description = "abc"
    inputs = ["in2_1", "in2_2", "out1_2"]
    outputs = ["out2_1", "out2_2", "out2_3"]

    def run(self):
        print("Do something")


class Stage3:
    name = "stage3"
    description = "abc"
    inputs = ["out1_2", "out2_3", "out1_3", "out2_2"]
    outputs = ["out3_1", "out3_2", "out3_3"]

    def run(self):
        print("Do something")