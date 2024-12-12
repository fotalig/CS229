class parent_class(object):

    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def get_var1(self):
        return self.var1

    def get_var2(self):
        return self.var2

    def set_var1(self, var1_new):
        self.var1 = var1_new

    def set_var2(self, var2_new):
        self.var2 = var2_new

class child_class(parent_class):

    def var_addition(self):
        return self.var1 + self.var2


test_obj = child_class(1,2)

addition_test = test_obj.var_addition()

print(test_obj, addition_test)