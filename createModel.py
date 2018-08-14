from vecNorm import vecNorm
from HDM import *
from hdmNgram import hdmNgram
from hdmUOG import hdmUOG
from normalVector import normalVector
from readCorpus import readCorpus
import os
import logging

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

"""READCORPUS reads a corpus into a multi-layer HDM
   N is the dimensionality of the vectors
   L is the number of layers
   W is the window size, placeholder+- W, set to zero to use full sentence

   MATRIX is the corpus of indices
       each row is a sentence
       each word/cell is an index
       readCorpus function returns this matrix

   LABELS are the word labels generated
       using the word2index.m function
       readCorpus function returns labels
"""

def createModels( Dim, Layer, Win, matrix, labels, left,group,placeholder,percepts ):
    
    # PARAMETER DEFAULTS
    if Dim is None:
        Dim = 256 # dimensionality

    if Layer is None:
        Layer = 1 # layers

    if Win is None:
        Win = 0  # window size, by default use entire sentence / no window

                # TEST DATA SET: If no parameters are given, demo with test data se
      #  TEST_MODE = False
    if labels is None or matrix is None:
        labels = np.array(['cheatword ','eagles ','birds ','airplanes ','dishes ',
                               'squirrels ','soar ','fly ','drive ','are ',
                               'live ','over ','above ','through ','on ','in ',
                               'trees ','forest ','skies ','plates ', 'streets ','cars '])
        matrix = np.array([[4, 1,  6, 11, 16], # eagles soar over trees  (first index shows the length of the sentence)
                               [4, 2,  7, 12, 17], # birds fly above forest
                               [4, 3,  6, 13, 18], # airplanes soar through skies
                               [4, 3,  7, 13, 18] , # airplanes fly through skies
                               [4, 4,  9, 11, 19], # dishes are over plates
                               [4, 4, 9, 12, 19], # dishes are above plates
                               [4, 5, 10, 15, 16], # squirrels live in trees
                               [4, 5, 10, 15, 17], # squirrels live in forest
                               [4, 21,  8, 14, 20]]) # cars drive on streets
                          #     [4,  9, 14, 19]) # dishes are atop plates




    # HIDDEN PARAMETERS
    M = np.shape(labels)[0] # number of items in memory
    NORMALIZE = True # normalize between layers (RECOMMENDED)
    SKIPGRAM  = False  # use skip grams rather than conventional n-grams
    NGRAMTEST = False
    savedir = 'models'


    #CONSTRUCT MODEL
    if left is None:
        left  = getShuffle(N)
        np.savetxt(os.path.join(savedir,'left'), left,fmt='%i')
    if group is None:
        group = getShuffle(N)
        np.savetxt(os.path.join(savedir, 'group'), group, fmt='%i')
    if placeholder is None:
        placeholder = normalVector(N)
        np.savetxt(os.path.join(savedir, 'placeholder'), placeholder)

    #first layer uses random vectors as percepts
    if percepts is None:
        percepts = np.zeros((M,N))
        for i in range (0,M):
            percepts[i] = normalVector(N)
        np.savetxt(os.path.join(savedir, 'percepts'), percepts)
    #create permutations
    
    models = []
    emodels = []
    #cosines = zeros((M,M,L))
    for h in range(0,Layer):
        logging.info("Training model %s"%(h+1))
        name = 'L'+str(h+1)+'W'+str(Win)
        model = HDM(percepts,percepts,labels,name,placeholder,left)
        for s,sent in enumerate(matrix):
            s = s+1
            if s%1000==0:  # To show the progress
                logging.info (s)
              # check to see if there's a invalid index:
               #   disallow Inf, NaN, and Zero values
        #    foundMin=np.where( matrix[s] == np.inf or matrix[s] == -np.inf )[0]
                # if there is an invalid index, end the sentence before it
        #    if len(foundMin)!=0:
        #        sent_len = foundMin[0] - 1
        #    else: # otherwise, the sentence is the maximum length
            sent_len = sent[0] #MAX_WORDS
            #print sent_len
            for r in range(0,sent_len):
                    # set window boundaries
                if Win > 0:  # read data within a sentence using a window ##SHOULD BE FIXED
                    first  = max(1,r-Win)
                    last   = min(r+Win+1,sent[0])
                    target = r - first +1
                else: #don't use a window if W = 0
                    first  = 1
                    last = sent_len
                    target = r
                    # construct window
                indice = sent[first:last+1]
                t = sent[target+1]
                try:
                    window = percepts[np.array(indice)]
                except:
                    print(s,sent)
                window[target] = placeholder
                #print "place ",placeholder
            #    if r==1 and h==0:
            #        print percepts[[1,2]]
            #        print "window" , s , ": ", window
                    # construct n-gram or skip-gram and add it to model
                if SKIPGRAM:
                    experience = hdmUOG(window,target,left)
                else:
                    experience = hdmNgram(window,target,left)
                    #if h==0 and r==1:
                        #print s, window#, target, left,experience
                model.HDMadd(int(sent[r+1]),experience) ## +1 because the first word is at index 1, and the last is at index sent_len.
        # use concepts to construct percepts for next layer
        writeModel(h,model,savedir)

        if h < Layer:
        # permute them to protect the data
            percepts = copy(model.concepts[:,group])
            # normalize if you don't want to bias representations
            # such that they become dominated by the most familiar items
            if NORMALIZE:
                for i in range(0,M):
                    percepts[i] = vecNorm(copy(percepts[i]))
    # return models, emodels
# DISPLAY VISUALS IF IN TEST MODE

def writeModel(h,model,dir):
    level = h + 1
    if not os.path.exists(dir):
        os.mkdir(dir)

    logging.info('saving the level %s model'%(level))
    mmodelpath = os.path.join(dir, 'beagle' + model.name+'.m')
    emodelpath = os.path.join(dir, 'beagle' + model.name+'.e')

    with open(mmodelpath, 'w') as tf:
        tf.write(str(len(labels)) + ' ' + str(N) + '\n')
        for w in range(1, len(labels)):
            word = model.labels[w]
            vector = model.concepts[w]

            tf.write(word + ' ' + " ".join([str(num) for num in vector]) + '\n')

    with open(emodelpath, 'w') as tf:
        tf.write(str(len(labels)) + ' ' + str(N) + '\n')
        for w in range(1, len(labels)):
            word = model.labels[w]
            vector = model.percepts[w]

            tf.write(word + ' ' + " ".join([str(num) for num in vector]) + '\n')



## Read the corpus
labels, matrix=readCorpus('data',5)
N = 1024

## If you want to use specific values for left,placeholder, group, or percepts
#left, placeholder,group, percepts=test()

## Creating the models
#createModels(Dim, Layer, Win, matrix, labels, left,group,placeholder,percepts )
createModels(N,3,5,copy(np.array(matrix)),copy(labels),None,None,None,None)#left,group,placeholder,None)


        
""" FOR TESTING A FEW WORDS ON SWITHCBOARD CORPUS

dogs=np.array([])
cats=np.array([])
within=np.array([])

with open('model1Concepts.csv', 'w') as csvoutput:
    writer = csv.writer(csvoutput)
    words=[2733,3234,3247]
    for i in range (0,len(labels)):
        if i==2733:
            dogs=models[0].concepts[i]
        if i==3234:
            cats=models[0].concepts[i]
        if i==3247:
            within=models[0].concepts[i]
        writer.writerow([models[0].labels[i]]+[models[0].concepts[i]])

    
with open('testResults.csv', 'w') as csvoutput:
    writer = csv.writer(csvoutput)
    writer.writerow(dogs)
    writer.writerow(cats)
    writer.writerow(within)
    writer.writerow(["dogs"]+["cats"]+ [vectorCosine(cats,dogs)])
    writer.writerow(["dogs"]+["within"]+ [vectorCosine(dogs,within)])
    writer.writerow(["within"]+["cats"]+ [vectorCosine(within,cats)])
            
        
"""


# Create a Matrix of cosines between each words (only for the first level, you can set the level)

""" 
level=0
with open('results.csv', 'w') as csvoutput:
    writer = csv.writer(csvoutput)
    writer.writerow(models[level].labels)
    for i in range (0,len(labels)):
        cosines=[]
        for j in range (0,len(labels)): 
            cosines.append(vectorCosine(models[level].concepts[i], models[level].concepts[j]))
        writer.writerow([[models[level].labels[i]]+cosines)

"""

        
"""
with open('results.csv', 'w') as csvoutput:
    writer = csv.writer(csvoutput)
    writer.writerow(["word1"]+["word2"]+["model1"]+["model2"]+["model3"]+["model4"])
    for j in range (0,len(lables)):
        for k in range (j+1,len(labels)):
            writer.writerow([models[0].labels[j]]+[models[0].labels[k]] +[vectorCosine(models[0].concepts[j], models[0].concepts[k])]+[vectorCosine(models[1].concepts[j], models[1].concepts[k])]+[vectorCosine(models[2].concepts[j], models[2].concepts[k])]+[vectorCosine(models[3].concepts[j], models[3].concepts[k])])

"""
