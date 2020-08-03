# Define a class to receive the characteristics of each line detection
class line():
    def __init__(self):
        self.left_fitx = 0
        self.left_curverad =[]
        self.leftlinecoeff =0
        self.past_good_left_lines =[]

        self.right_fitx = 0
        self.right_curverad =[]
        self.rightlinecoeff =0
        self.past_good_right_lines=[]
        self.running_mean_difference_between_lines=0