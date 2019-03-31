### build custom models

Custom models can be implemented by inheriting the BaseMode and implementing the following three
methods: **_build_graph**, **compile**,and **get_best_answer**. A simple example is shown in the following code.
```python
from sogou_mrc.model.base_model import BaseModel
class CustomModel(BaseModel):
    def __init__(vocab, other_params)
        super(BertBaseline, self).__init__(vocab)
        self._build_graph()
    def _build_graph():

        '''
        The following variables should be defined in custom models.
        1.The input placeholder dict is used in the trainer to obtain the corresponding field in each batch data.
            self.input_placeholder_dict = xxx.
        2.output_variable_dict：the values of the variables defined in this dict can be obtained after evaluation
           self.output_variable_dict = xxx

        '''
        #caculate loss
        self.loss =....
    #define the optimizer and train_op
    def compile():
        self.train_op = optimizer.minimize(self.loss)
    '''
    Args：
        output： variables of model defined in the output_variable_dict e.g : probability, logits
        instances: each instance has a corresponding result in the output
    For evaluation:
        The result of each instances should be fed into the methods (to obtain the score) defined in corresponding evaluator.

    '''
    def get_best_answer(self,output, instances,other_params):

        return xxx

````

