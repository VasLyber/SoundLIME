'''
Created on 4 Feb 2017

@author: Saumitra

Edits:
21 Apr 2019 - Shreyan

'''
"""
Some code sections in this file are take
from Jan Schluter's ismir2015 SVD code and Lasagne saliency
map recipe.
For more details check:
https://github.com/f0k/ismir2015
https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
"""

import io

from optparse import OptionParser
import librosa
import numpy.ma as ma
import numpy.linalg as linalg
import sys
sys.path.append("..")

from slime import lime_image

lime_anal = True

from imports_and_utils import *

midlevel_dict = {
    0:'melody',
    1:'articulation',
    2:'r_complexity',
    3:'r_stability',
    4:'dissonance',
    5:'tonal stability',
    6:'minorness'
}
rState = np.random.RandomState(seed=0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def opts_parser():
    usage = \
        """\r%prog: Computes predictions with a neural network trained for singing
        voice detection.
        
        Usage: %prog [OPTIONS] MODELFILE OUTFILE
          MODELFILE: file to load the learned weights from (.npz format)
          OUTFILE: file to save the prediction curves to (.npz format)  
        """
    parser = OptionParser(usage=usage)
    parser.add_option('--dataset',
                      type='str', default='jamendo',
                      help='Name of the dataset to use.')
    parser.add_option('--pitchshift', metavar='PERCENT',
                      type='float', default=0.0,
                      help='If given, perform test-time pitch-shifting of given amount '
                           'and direction in percent (e.g., -10 shifts down by 10%).')
    parser.add_option('--mem-use',
                      type='choice', choices=('high', 'mid', 'low'), default='low',
                      help='How much temporary memory to use. More memory allows a '
                           'faster implementation, applying the network as a fully-'
                           'convolutional net to longer excerpts or the full files.')
    parser.add_option('--cache-spectra', metavar='DIR',
                      type='str', default=None,
                      help='Store spectra in the given directory (disabled by default).')
    parser.add_option('--plot',
                      action='store_true', default=False,
                      help='If given, plot each spectrogram with predictions on screen.')

    # new command line options. added for reading and saving partial file.
    parser.add_option('--partial',
                      action='store_true', default=False,
                      help='If given, read and predict the audio file only for the given duration and offset.')
    parser.add_option('--offset',
                      type='float', default=0.0,
                      help='read from the given offset location in the file for partial file read case.')
    parser.add_option('--duration',
                      type='float', default=3.2,
                      help='read for the given duration from the file.')
    parser.add_option('--transform', type='choice', default='mel', choices=('mel', 'spect'),
                      help='decides the dimensions of input fed to the neural network')
    parser.add_option('--save_input',
                      action='store_true', default=False,
                      help='if given stores the read input audio and the extracted transform.')
    parser.add_option('--dump_path', metavar='DIR',
                      type='str', default=None,
                      help='Store important analysis information in given directory (disabled by default).')
    return parser



def prepare_audio(specpath):
    """
    loads pre-prepared spectrogram

    """
    x = SpecRegressionDataPool_noTargets([specpath], return_start_stop_times=True, seed=rState)
    # first dimension is 0 because we want the spectrum of a single audio (specpath is a single file)
    spec_10sec = x[0][0][0]
    file_list = x[0][1][0]
    start_stop_times = x[0][2][0]
    return spec_10sec, start_stop_times


def _predict_joint(model, device, test_data):

    model.eval()
    pred_list_ml = []
    pred_list_emo = []
    num_clips = test_data.shape[0]

    with torch.no_grad():
        for clip_idx in range(num_clips):
            inputs = test_data[clip_idx:((clip_idx + 1))]
            inputs = torch.Tensor(inputs).to(device)

            inputs = inputs.unsqueeze(1)

            output_ml, output_emo = model(inputs)

            pred_list_ml.append(output_ml.cpu().numpy())
            pred_list_emo.append(output_emo.cpu().numpy())

    pred_list_ml = np.vstack(pred_list_ml)
    pred_list_emo = np.vstack(pred_list_emo)

    return [pred_list_ml, pred_list_emo]

'''
def compile_prediction_function_audio(modelfile):
    """
    Compiles a function to compute the classification prediction
    for a given number of input excerpts.
    """
    # instantiate neural network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(audio2mlemo_model,
                       path=modelfile,
                       device=device,
                       num_targets=8)

    pred_func = _predict_joint
    args = [model, device]

    return pred_func, args
'''

def compile_prediction_function_audio(modelfile):
    """
    Compiles a function to compute the classification prediction
    for a given number of input excerpts.
    """
    # instantiate neural network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(audio2mlemo_model,
                       path=modelfile,
                       device=device,
                       num_targets=8)

    def pred_func(x):
        return _predict_joint(model, device, x)

    return pred_func



if __name__ == '__main__':

    # parse command line
    parser = opts_parser()
    options, args = parser.parse_args()
    if len(args) < 2:
        parser.error("missing MODELFILE and/ or SEED")
    (modelfile, seed) = args
    seed = int(seed)
    rState = np.random.RandomState(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # Generate excerpts from input audio
    # returns a "list" of 3d arrays where each element has shape (no. of excerpts) x 115 x 80
    spectrum, start_stop_times = prepare_audio('/home/shreyan/mounts/home@rk0/PROJECTS/midlevel/Soundtracks/set1/set1/mp3/spec/045.mp3.spec')

    from skimage.segmentation import slic, felzenszwalb, mark_boundaries

    segments = felzenszwalb(spectrum / np.max(np.abs(spectrum)), scale=25, min_size=40)
    plt.imshow(np.rot90(mark_boundaries(spectrum / np.max(np.abs(spectrum)), segments, mode='subpixel')))
    plt.title("Segments")
    plt.show()
    plt.imshow(np.rot90(spectrum))
    plt.xticks(np.linspace(0,spectrum.shape[0], 5).astype(int), np.linspace(start_stop_times[0], start_stop_times[1], 5).round(1))
    plt.show()



    # compile the prediction function
    print('Compiling CNN prediction function ....')
    prediction_fn_audio = compile_prediction_function_audio(modelfile)

    ############################LIME/SLIME-BASED ANALYSIS#############################
    # We know apply SLIME to the CNN model to generate time-frequency based explanations.
    list_exp = []

    print("\n------LIME based analysis-----")
    explainer = lime_image.LimeImageExplainer(verbose=True)
    explanation, seg = explainer.explain_instance(image=spectrum, classifier_fn=prediction_fn_audio, hide_color=0,
                                                  top_labels=5, num_samples=2000, seed=rState)

    for i in range(7):
        temp, mask, fs = explanation.get_image_and_mask(i, positive_only=True, hide_rest=True, num_features=30)

        plt.imshow(np.rot90(temp))
        plt.xticks(np.linspace(0,spectrum.shape[0], 5).astype(int), np.linspace(start_stop_times[0], start_stop_times[1], 5).round(1))
        plt.title(midlevel_dict[i])
        plt.show()
        pass






