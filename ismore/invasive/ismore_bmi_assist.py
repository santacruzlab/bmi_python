from riglib.bmi.assist import FeedbackControllerAssist

class OrthoAssistOnly(EndptOrthoAssist):
    '''
    Summary: An assister to be used to dampen motion of the armassist and rehand
        orthogonal to the calculated intended trajectory
    '''
    def __init__(self, drives_plant, *args, **kwargs):
        self.drives_plant = drives_plant
        
    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        #Calculate typical x_assist component: 
        x_assist = self.fb_ctrl.calc_next_state(current_state, target_state, mode=mode)
        
        #Project current_state on normalized x_assist: 
        vect = x_assist[self.drives_plant]/np.linalg.norm(x_assist[self.drives_plant])
        proj = np.dot(current_state[self.drives_plant], vect)

        #Now subtract projection: 
        orth = current_state[self.drives_plant] - proj

        x_assist = calc_combined_assist(x_assist, assist_level, orth)
        return dict(x_assist=x_assist, assist_level=assist_level)

    def calc_combined_assist(x_assist, assist_level, orth):
        return -1*assist_level*orth

class EndptOrthoAssist(OrthoAssistOnly):

    def calc_combined_assist(x_assist, assist_level, orth):
        return x_assist - (assist_level*orth)


