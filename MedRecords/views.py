from __future__ import division
from __future__ import print_function
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from collections import Counter
import random
import numpy as np
import cv2
import sys
import numpy as np
import tensorflow as tf
import os
import argparse
import editdistance
import re


# Create your views here.
def index(request):
    return render(request, 'MedRecords/index.html', {})

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)
    last_process()

    
        
        
        
        
    return render(request, 'MedRecords/upload.html', {})


def last_process():
    
    def words(text): return re.findall(r'\w+', text.lower())

    WORDS = Counter(words(open('media/words.txt').read()))

    def P(word, N=sum(WORDS.values())): 
        "Probability of `word`."
        return WORDS[word] / N

    def correction(word): 
        "Most probable spelling correction for word."
        return max(candidates(word), key=P)

    def candidates(word): 
        "Generate possible spelling corrections for word."
        return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

    def known(words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in WORDS)

    def edits1(word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in edits1(word) for e2 in edits1(e1))

    def preprocess(img, imgSize, dataAugmentation=False):
        "put img into target img of size imgSize, transpose for TF and normalize gray-values"


        if img is None:
            img = np.zeros([imgSize[1], imgSize[0]])


        if dataAugmentation:
            stretch = (random.random() - 0.5) # -0.5 .. +0.5
            wStretched = max(int(img.shape[1] * (1 + stretch)), 1) # random width, but at least 1
            img = cv2.resize(img, (wStretched, img.shape[0])) # stretch horizontally by factor 0.5 .. 1.5


        (wt, ht) = imgSize
        (h, w) = img.shape
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
        img = cv2.resize(img, newSize)
        target = np.ones([ht, wt]) * 255
        target[0:newSize[1], 0:newSize[0]] = img

        # transpose for TF
        img = cv2.transpose(target)

        # normalize
        (m, s) = cv2.meanStdDev(img)
        m = m[0][0]
        s = s[0][0]
        img = img - m
        img = img / s if s>0 else img
        return img

    class Sample:
        "sample from the dataset"
        def __init__(self, gtText, filePath):
            self.gtText = gtText
            self.filePath = filePath


    class Batch:
        "batch containing images and ground truth texts"
        def __init__(self, gtTexts, imgs):
            self.imgs = np.stack(imgs, axis=0)
            self.gtTexts = gtTexts


    class DataLoader:
        "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database" 

        def __init__(self, filePath, batchSize, imgSize, maxTextLen):
            "loader for dataset at given location, preprocess images and text according to parameters"

            assert filePath[-1]=='/'

            self.dataAugmentation = False
            self.currIdx = 0
            self.batchSize = batchSize
            self.imgSize = imgSize
            self.samples = []

            f=open(filePath+'sentences.txt')
            chars = set()
            bad_samples = []
            bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
            for line in f:

                if not line or line[0]=='#':
                    continue

                lineSplit = line.strip().split(' ')
                assert len(lineSplit) >= 9


                fileNameSplit = lineSplit[0].split('-')
                fileName = filePath + 'sentences/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

                # GT text are columns starting at 9
                gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
                chars = chars.union(set(list(gtText)))

                # check if image is not empty
                if not os.path.getsize(fileName):
                    bad_samples.append(lineSplit[0] + '.png')
                    continue

                # put sample into list
                self.samples.append(Sample(gtText, fileName))

            # some images in the IAM dataset are known to be damaged, don't show warning for them
            if set(bad_samples) != set(bad_samples_reference):
                print("Warning, damaged images found:", bad_samples)
                print("Damaged images expected:", bad_samples_reference)

            # split into training and validation set: 95% - 5%
            splitIdx = int(0.95 * len(self.samples))
            self.trainSamples = self.samples[:splitIdx]
            self.validationSamples = self.samples[splitIdx:]

            # put words into lists
            self.trainWords = [x.gtText for x in self.trainSamples]
            self.validationWords = [x.gtText for x in self.validationSamples]

            # number of randomly chosen samples per epoch for training 
            self.numTrainSamplesPerEpoch = 25000 

            # start with train set
            self.trainSet()

            # list of all chars in dataset
            self.charList = sorted(list(chars))


        def truncateLabel(self, text, maxTextLen):

            cost = 0
            for i in range(len(text)):
                if i != 0 and text[i] == text[i-1]:
                    cost += 2
                else:
                    cost += 1
                if cost > maxTextLen:
                    return text[:i]
            return text


        def trainSet(self):
            "switch to randomly chosen subset of training set"
            self.dataAugmentation = True
            self.currIdx = 0
            random.shuffle(self.trainSamples)
            self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]


        def validationSet(self):
            "switch to validation set"
            self.dataAugmentation = False
            self.currIdx = 0
            self.samples = self.validationSamples


        def getIteratorInfo(self):
            "current batch index and overall number of batches"
            return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


        def hasNext(self):
            "iterator"
            return self.currIdx + self.batchSize <= len(self.samples)


        def getNext(self):
            "iterator"
            batchRange = range(self.currIdx, self.currIdx + self.batchSize)
            gtTexts = [self.samples[i].gtText for i in batchRange]
            imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
            self.currIdx += self.batchSize
            return Batch(gtTexts, imgs)

    class DecoderType:
        BestPath = 0
        BeamSearch = 1
        WordBeamSearch = 2


    class Model: 
        "minimalistic TF model for HTR"

        # model constants
        batchSize = 20
        imgSize = (128,32)
        maxTextLen = 32

        def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
            "init model: add CNN, RNN and CTC and initialize TF"
            self.charList = charList
            self.decoderType = decoderType
            self.mustRestore = mustRestore
            self.snapID = 0

            # Whether to use normalization over a batch or a population
            self.is_train = tf.placeholder(tf.bool, name="is_train");

            # input image batch
            self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

            # setup CNN, RNN and CTC
            self.setupCNN()
            self.setupRNN()
            self.setupCTC()

            # setup optimizer to train NN
            self.batchesTrained = 0
            self.learningRate = tf.placeholder(tf.float32, shape=[])
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
            with tf.control_dependencies(self.update_ops):
                self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

            # initialize TF
            (self.sess, self.saver) = self.setupTF()


        def setupCNN(self):
            "create CNN layers and return output of these layers"
            cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

            # list of parameters for the layers
            kernelVals = [5, 5, 3, 3, 3]
            featureVals = [1, 32, 64, 128, 128, 256]
            strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
            numLayers = len(strideVals)

            # create layers
            pool = cnnIn4d # input to first CNN layer
            for i in range(numLayers):
                kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
                conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
                conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
                relu = tf.nn.relu(conv_norm)
                pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

            self.cnnOut4d = pool


        def setupRNN(self):
            "create RNN layers and return output of these layers"
            rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

            # basic cells which is used to build RNN
            numHidden = 256
            cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

            # stack basic cells
            stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            # bidirectional RNN
            # BxTxF -> BxTx2H
            ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)

            # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
            concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

            # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
            kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
            self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])


        def setupCTC(self):
            "create CTC loss and decoder and return them"
            # BxTxC -> TxBxC
            self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
            # ground truth text as sparse tensor
            self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

            # calc loss for batch
            self.seqLen = tf.placeholder(tf.int32, [None])
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

            # calc loss for each element to compute label probability
            self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
            self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

            # decoder: either best path decoding or beam search decoding
            if self.decoderType == DecoderType.BestPath:
                self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
            elif self.decoderType == DecoderType.BeamSearch:
                self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
            elif self.decoderType == DecoderType.WordBeamSearch:

                word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

                # prepare information about language (dictionary, characters in dataset, characters forming words) 
                chars = str().join(self.charList)
                wordChars = open('model/wordCharList.txt').read().splitlines()[0]
                corpus = open('data/corpus.txt').read()

                # decode using the "Words" mode of word beam search
                self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))


        def setupTF(self):
            "initialize TF"
            print('Python: '+sys.version)
            print('Tensorflow: '+tf.__version__)

            sess=tf.Session() # TF session

            saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
            modelDir = 'model/'
            latestSnapshot = tf.train.latest_checkpoint(modelDir)
            file_writer = tf.summary.FileWriter('/path/to/logs', sess.graph) # is there a saved model?

            # if model must be restored (for inference), there must be a snapshot
            if self.mustRestore and not latestSnapshot:
                raise Exception('No saved model found in: ' + modelDir)

            # load saved model if available
            if latestSnapshot:
                print('Init with stored values from ' + latestSnapshot)
                saver.restore(sess, latestSnapshot)
            else:
                print('Init with new values')
                sess.run(tf.global_variables_initializer())

            return (sess,saver)


        def toSparse(self, texts):
            "put ground truth texts into sparse tensor for ctc_loss"
            indices = []
            values = []
            shape = [len(texts), 0] # last entry must be max(labelList[i])

            # go over all texts
            for (batchElement, text) in enumerate(texts):
                # convert to string of label (i.e. class-ids)
                labelStr = [self.charList.index(c) for c in text]
                # sparse tensor must have size of max. label-string
                if len(labelStr) > shape[1]:
                    shape[1] = len(labelStr)
                # put each label into sparse tensor
                for (i, label) in enumerate(labelStr):
                    indices.append([batchElement, i])
                    values.append(label)

            return (indices, values, shape)


        def decoderOutputToText(self, ctcOutput, batchSize):
            "extract texts from output of CTC decoder"

            # contains string of labels for each batch element
            encodedLabelStrs = [[] for i in range(batchSize)]

            # word beam search: label strings terminated by blank
            if self.decoderType == DecoderType.WordBeamSearch:
                blank=len(self.charList)
                for b in range(batchSize):
                    for label in ctcOutput[b]:
                        if label==blank:
                            break
                        encodedLabelStrs[b].append(label)

            # TF decoders: label strings are contained in sparse tensor
            else:
                # ctc returns tuple, first element is SparseTensor 
                decoded=ctcOutput[0][0] 

                # go over all indices and save mapping: batch -> values
                idxDict = { b : [] for b in range(batchSize) }
                for (idx, idx2d) in enumerate(decoded.indices):
                    label = decoded.values[idx]
                    batchElement = idx2d[0] # index according to [b,t]
                    encodedLabelStrs[batchElement].append(label)

            # map labels to chars for all batch elements
            return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


        def trainBatch(self, batch):
            "feed a batch into the NN to train it"
            numBatchElements = len(batch.imgs)
            sparse = self.toSparse(batch.gtTexts)
            rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
            evalList = [self.optimizer, self.loss]
            feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True}
            (_, lossVal) = self.sess.run(evalList, feedDict)
            self.batchesTrained += 1
            return lossVal


        def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
            "feed a batch into the NN to recognize the texts"

            # decode, optionally save RNN output
            numBatchElements = len(batch.imgs)
            evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])
            feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
            evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], feedDict)
            decoded = evalRes[0]
            texts = self.decoderOutputToText(decoded, numBatchElements)

            # feed RNN output and recognized text into CTC loss to compute labeling probability
            probs = None
            if calcProbability:
                sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
                ctcInput = evalRes[1]
                evalList = self.lossPerElement
                feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
                lossVals = self.sess.run(evalList, feedDict)
                probs = np.exp(-lossVals)
            return (texts, probs)


        def save(self):
            "save model to file"
            self.snapID += 1
            self.saver.save(self.sess, 'model/snapshot', global_step=self.snapID)


    class FilePaths:
        "filenames and paths to data"
        fnCharList = 'model/charList.txt'
        fnAccuracy = 'model/accuracy.txt'
        fnTrain = 'data/'
        fnInfer = 'data/test.png'
        fnCorpus = 'data/corpus.txt'
        newPath = 'data/new.png'


    def train(model, loader):
        "train NN"
        epoch = 0 # number of training epochs since start
        bestCharErrorRate = float('inf') # best valdiation character error rate
        noImprovementSince = 0 # number of epochs no improvement of character error rate occured
        earlyStopping = 5 # stop training after this number of epochs without improvement
        while True:
            epoch += 1
            print('Epoch:', epoch)

            # train
            print('Train NN')
            loader.trainSet()
            while loader.hasNext():
                iterInfo = loader.getIteratorInfo()
                batch = loader.getNext()
                loss = model.trainBatch(batch)
                print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

            # validate
            charErrorRate = validate(model, loader)

            # if best validation accuracy so far, save model parameters
            if charErrorRate < bestCharErrorRate:
                print('Character error rate improved, save model')
                bestCharErrorRate = charErrorRate
                noImprovementSince = 0
                model.save()
                #open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
            else:
                print('Character error rate not improved')
                noImprovementSince += 1

            # stop training if no more improvement in the last x epochs
            if noImprovementSince >= earlyStopping:
                print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
                break


    def validate(model, loader):
        "validate NN"
        print('Validate NN')
        loader.validationSet()
        numCharErr = 0
        numCharTotal = 0
        numWordOK = 0
        numWordTotal = 0
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            print('Batch:', iterInfo[0],'/', iterInfo[1])
            batch = loader.getNext()
            (recognized, _) = model.inferBatch(batch)

            print('Ground truth -> Recognized')	
            for i in range(len(recognized)):
                numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
                numWordTotal += 1
                dist = editdistance.eval(recognized[i], batch.gtTexts[i])
                numCharErr += dist
                numCharTotal += len(batch.gtTexts[i])
                print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

        # print validation result
        charErrorRate = numCharErr / numCharTotal
        wordAccuracy = numWordOK / numWordTotal
        #print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
        return charErrorRate

    shortform = {
        "#" : "fracture/cycle",
        "+" : "present",
        "a/w" : "associated with",
        "AVM" : "Artnio Venous Malformation",
        "MRI" : "Magnetic  Resonance imaging",
        "CT" : "Computed tomography",
        "USG" : "Ultra Sonography",
        "KUB" : "Kidney Ureter bladder",
        "ADV" : "Adviced",
        "B/L" : "Bilateral",
        "BBF" : "before breakfast",
        "BD" : "before dinner",
        "BID" : "two times a day",
        "BL" : "before lunch",
        "BP" : "Blood pressure",
        "Bx" : "Biospy",
        "BSO" : "Bilateral Salphingo Oophorectomy",
        "c/o" : "complaint of",
        "CBC" : "Complete Blood Count",
        "CNS" : "Central vervous system",
        "ct" : "Continue",
        "CVS" : "Cardovascular System",
        "CXR" : "Chest x-ray",
        "CxR" : "Chest x-ray",
        "DAMA" : "Discharge against medical advice",
        "DN" : "Diabetic neuropathy",
        "DNB" : "Diabetic nephropathy",
        "DOR" : "Discharge on request",
        "DR" : "Diabetic retinopathy",
        "DM" : "Diabetis Mellitty",
        "E/O" : "evidence of",
        "F/H" : "Family history",
        "F/u" : "Follow up",
        "FTND" : "full term normal delivery",
        "G/E" : "General Examination",
        "GA" : "general anesthesia",
        "GC" : "General condition",
        "H/o" : "History of",
        "HP" : "Histopathalogy",
        "HPE" : "Histopathalogy",
        "HS" : "at bed time",
        "HTN" : "hypertension",
        "HPR" : "Hyptoglobin related protein",
        "POD" : "Port operative days",
        "I/v/o" : "in view of",
        "Inv" : "Investigation",
        "k/w/o" : "known case of",
        "L/E" : "local examination",
        "LE" : "left eye",
        "LA" : "Local application",
        "LA" : "Local Anesthesia",
        "LVSI" : "Lymphovascular space invasion",
        "NAD" : "No abnormality detected",
        "NBA" : "nil by mouth",
        "NED" : "No evidence of disease",
        "O/E" : "On examination",
        "O/H" : "Obsteric History",
        "OH" : "past history",
        "OU" : "both ears",
        "P" : "Pulse",
        "P/A" : "Per Abdomen",
        "PNA" : "patient not brought",
        "PO" : "per oral",
        "P/O" : "per oral",
        "POD" : "post operation day",
        "PR" : "Pulse rate",
        "PR" : "Per rectum",
        "PV" : "per vaginal",
        "r/o" : "rule out",
        "RR" : "Respiratory Rate",
        "S/b" : "Seen by",
        "s/c" : "subcutaneous",
        "S/E" : "Systemic Examination",
        "S/L" : "Sublingual",
        "s/o" : "suggestive of",
        "s/p" : "status post",
        "SA" : "Spinal anesthesia",
        "SOS" : "As required", 
        "T/T" : "Treatment given",
        "TAH" : "Trans Abdominal hysterectony",
        "Rx" : "Treatment given",
        "TID" : "thrice a day",
        "TSR" : "To show report",
        "w/c" : "without",
        "w/h" : "withhold",
        "WNL" : "within normal limits",
    }


    def infer(model, fnImg):
        "recognize text in image provided by file path"
        img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
        batch = Batch(None, [img])
        (recognized, probability) = model.inferBatch(batch, True)
        #print('Recognized:', '"' + recognized[0] + '"')
        #print('Probability:', probability[0])
        if len(recognized[0]) < 5:
            if recognized[0] in shortform:
                return shortform[recognized[0]]
            else:
                return correction(recognized[0])
        else:
            return correction(recognized[0])



    def main():
        "main function"
        # optional command line args
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", help="train the NN", action="store_true")
        parser.add_argument("--validate", help="validate the NN", action="store_true")
        parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
        parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
        args = parser.parse_args()

        decoderType = DecoderType.BestPath
        if args.beamsearch:
            decoderType = DecoderType.BeamSearch
        elif args.wordbeamsearch:
            decoderType = DecoderType.WordBeamSearch


        if args.train or args.validate:
            # load training data, create TF model
            loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

            # save characters of model for inference mode
            open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

            # save words contained in dataset into file
            open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

            # execute training or validation
            if args.train:
                model = Model(loader.charList, decoderType)
                train(model, loader)
            elif args.validate:
                model = Model(loader.charList, decoderType, mustRestore=True)
                validate(model, loader)

        # infer text on test image
        else:
            print(open(FilePaths.fnAccuracy).read())
            model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
            list = os.listdir("images") # dir is your directory path
            number_files = len(list)
            print(number_files)
            f = open("predicted/predicted.txt","w+")

            for i in range(number_files):
                path1 = "images/" + str(i) + ".png"
                prediction = infer(model,path1)
                if i%10==0:
                    f.write("\n")
                    continue
                else:
                    f.write(prediction + " ")
                    continue
            f.close()
            print("Done!")

            #infer(model, FilePaths.fnInfer)
            #infer(model,FilePaths.newPath)




    if __name__ == '__main__':
        main()