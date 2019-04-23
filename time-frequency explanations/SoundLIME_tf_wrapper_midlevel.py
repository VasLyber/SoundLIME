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
    spec10sec = SpecRegressionDataPool_noTargets([specpath])
    return spec10sec[0][0][0]



def _predict_joint(model, device, test_data):

    model.eval()
    pred_list_ml = []
    pred_list_emo = []
    num_clips = test_data.shape[0]

    with torch.no_grad():
        for clip_idx in trange(num_clips, ascii=True):
            inputs, filenames = test_data[clip_idx:((clip_idx + 1))]
            inputs = torch.Tensor(inputs).to(device)

            inputs = inputs.unsqueeze(1)

            output_ml, output_emo = model(inputs)

            pred_list_ml.append(output_ml.cpu().numpy())
            pred_list_emo.append(output_emo.cpu().numpy())

    pred_list_ml = np.vstack(pred_list_ml)
    pred_list_emo = np.vstack(pred_list_emo)

    return [pred_list_ml, pred_list_emo]


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


    return model




if __name__ == '__main__':

    # parse command line
    parser = opts_parser()
    options, args = parser.parse_args()
    if len(args) < 2:
        parser.error("missing MODELFILE and/ or MEAN_STDFILE")
    (modelfile, meanstd_file) = args

    # default parameters from Jan Schluter et. al. paper @ ISMIR 2015
    frame_len = 1024
    nmels = 80
    excerpt_size = 115  # equivalent to 1.6 sec

    # dictionary of boolean flags to guide what needs to be the output file
    # all initialised to false initially
    # 'um' - unmasked, 'm'-masked, 'cm'- conditionally masked
    flags = dict.fromkeys(['um', 'm', 'cm'], False)

    if (options.transform == 'mel'):
        input_dim = nmels
    else:
        input_dim = (frame_len / 2) + 1


    # Generate excerpts from input audio
    # returns a "list" of 3d arrays where each element has shape (no. of excerpts) x 115 x 80
    spectrum = prepare_audio('/home/shreyan/mounts/home@rk0/PROJECTS/midlevel/Soundtracks/set1/set1/mp3/spec/001.mp3.spec')



    # compile the prediction function
    print('Compiling CNN prediction function ....')
    prediction_fn_audio = compile_prediction_function_audio(modelfile)

    ############################LIME/SLIME-BASED ANALYSIS#############################
    # We know apply SLIME to the CNN model to generate time-frequency based explanations.
    list_exp = []

    print("\n------LIME based analysis-----")
    explainer = lime_image.LimeImageExplainer(verbose=True)
    explanation, seg = explainer.explain_instance(image=spectrum, classifier_fn=prediction_fn_audio, hide_color=0,
                                                  top_labels=5, num_samples=2000)
    temp, mask, fs = explanation.get_image_and_mask(0, positive_only=True, hide_rest=True, num_features=3)
    print("Top-%d components in the explanation are: (%d, %d, %d)" % (3, fs[0][0], fs[0][1], fs[0][2]))

    list_exp.append(fs)  # if multiple explanations are generated for the same instance



