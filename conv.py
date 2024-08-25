# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        W = self.W

        input_size = np.shape(A)[-1]
        kernel_size = W.shape[-1]
        output_size = (input_size-kernel_size)+1
        out_channels = self.out_channels
        # print('output_size: ', output_size)
        # print('W shape: ', W.shape)
        # print('Kernel size: ', kernel_size)
        # print('------')

        Z = np.zeros((np.shape(A)[0], self.out_channels, output_size))
        convolution = np.zeros((self.in_channels, output_size))

        for ii in range(np.shape(Z)[0]):
          for jj in range(out_channels):
            for kk in range(self.in_channels):

              convolution[kk] = (np.convolve(A[ii][kk], np.flip(W[jj][kk]), mode='valid'))
            
            convolve_sum = np.sum(convolution,0)
            Z[ii][jj] = convolve_sum + self.b[jj]

        # for ii in range(np.shape(ZZ)[0]):
        #   for jj in range(out_channels):
        #     ZZ[ii][jj] = ZZ[ii][jj] + self.b[jj]

        # My debugging:
        # print('Shape ZZ: ', np.shape(ZZ))
        # print('outchannels: ', self.out_channels)
        # print('Bias: ', self.b)
        

        # Z = ZZ  # TODO
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size = np.shape(dLdZ)[0]
        out_channels = np.shape(dLdZ)[1]
        output_size = np.shape(dLdZ)[2]
        in_channels = np.shape(self.A)[1]


        # print('shape dLdZ: ', np.shape(dLdZ))
        # print('input channels: ', self.in_channels)
        # print('Input size: ', np.shape(self.A))
        # print('Output channels: ', out_channels)
        # print('Kernel size: ', self.kernel_size)
        # print('Batch size: ', batch_size)
        # print('size W: ', np.shape(self.W))

        # Broadcasting:
        dLdZ_broad = np.zeros((batch_size, out_channels, in_channels , output_size))
        # print('shape broad', np.shape(dLdZ_broad))
        for ii in range(batch_size):
          for jj in range(in_channels):
            for kk in range(out_channels):
              dLdZ_broad[ii][kk][jj] = dLdZ[ii][kk]

        # Convolving parameters:
        new_kernal_size = len(dLdZ[ii][kk])
        convolve_output_size = np.shape(self.A)[2] - new_kernal_size + 1

        DD = np.zeros((batch_size, out_channels, in_channels, convolve_output_size))

        # print('Size DD: ', np.shape(DD))

        for ii in range(batch_size):
          for jj in range(out_channels):
            for kk in range(in_channels):
              DD[ii][jj][kk] = np.convolve(self.A[ii][kk], np.flip(dLdZ_broad[ii][jj][kk]), mode='valid')

        self.dLdW = np.sum(DD,0) # TODO

        self.dLdb = np.sum(np.sum(dLdZ,2),0) # TODO

        convolve_output_size_DSP = np.shape(dLdZ)[2] + self.kernel_size - 1

        # print('convolve output size DSP: ', convolve_output_size_DSP)

        WW = np.zeros((batch_size, out_channels, in_channels, convolve_output_size_DSP))
        # print('Shape WW: ', np.shape(WW))
        for ii in range(batch_size):
          for jj in range(out_channels):
            for kk in range(in_channels):
              WW[ii][jj][kk] = np.convolve(dLdZ_broad[ii][jj][kk], self.W[jj][kk])




        dLdA = np.sum(WW,1) # TODO

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        # print('this is in_channels: ', in_channels)
        # print('this is out_channels: ', out_channels)
        # print('this is kernel_size: ', kernel_size)
        # print('this is the stride: ', stride)
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels,
        kernel_size, weight_init_fn, bias_init_fn) # TODO
        
        self.downsample1d = Downsample1d(stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        
        # Call Conv1d_stride1
        # TODO
        AA = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(AA) # TODO
        # print('this is the shape of input of conv1d.forward: ', A.shape)
        # print('this is the shape of AA in conv1d.forward: ', AA.shape)
        # print('this is the shape of output of conv1d.forward: ', Z.shape)
        

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        AA = self.downsample1d.backward(dLdZ)
        # print('this is the shape of dLdZ (input of conv1d.backward): ', dLdZ.shape)
        # print('this is the shape of AA:', AA.shape)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(AA) # TODO 

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)
            # self.W = weight_init_fn
        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)
            # self.b = bias_init_fn

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        AA = self.A
        WW = self.W
        # print('My WW: ', WW)
        bb = self.b
        # print('My bb: ', bb)
        # print('Shape AA: ', AA.shape)

        ##### fot the quiz
        # bb = np.zeros(3)

        in_channels = AA.shape[1]
        out_channels = WW.shape[0]
        kernel_size = WW.shape[-1]
        output_size_col = AA.shape[3]-kernel_size+1
        output_size_row = AA.shape[2]-kernel_size+1
        batch_size = AA.shape[0]

        GG = np.zeros((batch_size, in_channels, out_channels, output_size_row, output_size_col))
        difference_row = len(AA[0][0]) - len(WW[0][0])
        difference_col = len(AA[0][0][0]) - len(WW[0][0])

        for ii in range(batch_size):
          for jj in range(out_channels):
            for kk in range(in_channels):
              TT = AA[ii][kk]
              RR = WW[jj][kk]

              # print('Shape TT: ', TT.shape)
              # print('Shape RR: ', RR.shape)

              # difference_row = len(TT) - len(RR)
              # difference_col = len(TT[1]) - len(RR)
              # RR = np.pad(RR, ((0, difference_row), (0, difference_col)))
      
              # Convolution
              for mm in range(output_size_col):
                for nn in range(output_size_row):
                  # RR_new = np.roll(RR, mm, axis=0)
                  # RR_new = np.roll(RR_new, nn, axis=1)
                  # print("RR shape: ",RR.shape)
                  # print("TT shape: ", TT.shape)
                  TT_new = TT[mm:mm+kernel_size, nn:nn+kernel_size]
                  # print("TT_new shape: ", TT_new.shape)
                  GG[ii][kk][jj][mm][nn] = np.sum(RR*TT_new)
          # print('Here is GG: ', GG[ii])
    
        CC = np.sum(GG,1)
        out = np.zeros((batch_size, out_channels,output_size_row,output_size_col))
        for jj in range(batch_size):
          # print('batch iteration: ', ii)
          for ii in range(out_channels):
            out[jj][ii] = CC[jj][ii] + bb[ii]
            # print("here is CC: ", CC[jj])

        Z = out #TODO
        # print('Froward Done!')

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        AA = self.A
        W = self.W
        dLdZ_size = len(dLdZ[-1])
        # out_channels = dLdZ.shape[1]
        # in_channels = self.in_channels
        # batch_size = AA.shape[0]
        # kernel_size = self.kernel_size
        # output_size_col = AA.shape[3]-dLdZ_size+1
        # output_size_row = AA.shape[2]-dLdZ_size+1
        batch_size = dLdZ.shape[0]
        out_channels = W.shape[0]
        in_channels = W.shape[1]
        out_rows = dLdZ.shape[-2]
        out_cols = dLdZ.shape[-1]
        in_rows = AA.shape[-2]
        in_cols = AA.shape[-1]
        kernel_size = W.shape[-1]


        # prints:
        # print('batch_size: ', batch_size)
        # print('out channels: ', out_channels)
        # print('in channels: ', in_channels)
        # print('out rows: ', out_rows)
        # print('out cols: ', out_cols)
        # print('in rows: ', in_rows)
        # print('in cols: ', in_cols)
        # print('kernel size: ', kernel_size)



        ########################
        DLDW = np.zeros((batch_size, out_channels, in_channels, kernel_size, kernel_size))
        # print('dLdW shape: ', DLDW.shape)
        output_size_col = kernel_size
        output_size_row = kernel_size
        print('kernel_size: ', kernel_size)

        Pad_row = len(AA[0][0]) - len(dLdZ[0][0])
        Pad_col = len(AA[0][0][0]) - len(dLdZ[0][0][0])
        
        for ii in range(batch_size):
          for jj in range(in_channels):
            for kk in range(out_channels):
              
              TT = AA[ii][jj]
              RR = dLdZ[ii][kk]
              # print('This is TT shape before padding: ', TT.shape)
              # print('This is RR shape before padding: ', RR.shape)

              # Pad_row = len(TT) - len(RR)
              # Pad_col = len(TT[1]) - len(RR[1])

              
              
              # print('Here2: ', output_size_col)
              # print('Here3: ', output_size_row)


              # RR = np.pad(RR, ((0, Pad_row), (0, Pad_col)))
              # print('This is RR shape after padding: ', RR.shape)
              
              # convolution
              for mm in range(output_size_col):
                for nn in range(output_size_row):
                  RR_new = np.roll(RR, mm, axis=0)
                  RR_new = np.roll(RR_new, nn, axis=1)
                  
                  TT_new = TT[mm:mm+RR.shape[1], nn:nn+RR.shape[0]]
                  # print("RR shape: ", RR.shape)
                  # print("TT_new shape: ", TT_new.shape)
                  # print("RR shape: ", RR.shape)
                  DLDW[ii][kk][jj][nn][mm] = np.sum(RR*TT_new)

        DLDW = np.sum(DLDW,0) # TODO

        MM = DLDW

        for ii in range(DLDW.shape[0]):
          for jj in range(DLDW.shape[1]):
            DLDW[ii][jj] = np.transpose(MM[ii][jj])
        # print('dLdA shape: ', DLDW.shape)
        self.dLdW = DLDW
        ########################
        # delta=dLdZ
        # batch_size, out_channel, output_size = delta.shape
        # for batch in range(batch_size):
        #     for cOut in range(out_channels):
        #         for cIn in range(in_channels):
        #             for i in range(kernel_size):
        #                 for out in range(dLdZ_size):
        #                     self.dLdW[cOut, cIn, i] += self.A[batch, cIn, i + 1 * out] * delta[batch, cOut, out]
        # self.dLdW = self.dW
        ########################        
        





        DLDB = np.sum(dLdZ,0)
        DLDB = np.sum(DLDB,1)
        DLDB = np.sum(DLDB,1)
        self.dLdb = DLDB # TODO
        # print('in_channels: ', in_channels)
        # print('out channels: ', out_channels)
        # print('This is the dLdb: ', self.dLdb)
        
        padding = kernel_size-1 #padding for dLdZ
        # dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        DLDA = np.zeros((batch_size, out_channels, in_channels, in_rows, in_cols))

        
        for ii in range(batch_size):
          for jj in range(out_channels):
            for kk in range(in_channels):
              DLDZ = dLdZ[ii][jj]
              # DLDZ = np.sum(dLdZ,1)[ii]
              DLDZ = np.pad(DLDZ, ((padding,padding),(padding,padding)))
              myfilter = W[jj][kk]

              myfilter = np.flip(myfilter,axis=0)
              myfilter = np.flip(myfilter,axis=1)

              # Now convolution
              output_size_col = DLDZ.shape[1] - myfilter.shape[1] + 1
              output_size_row = DLDZ.shape[0] - myfilter.shape[0] + 1
              

              # Pad_row = len(DLDZ) - len(myfilter)
              # Pad_col = len(DLDZ[1]) - len(myfilter)

              # myfilter = np.pad(myfilter, ((0, Pad_row), (0, Pad_col)))

              for mm in range(output_size_col):
                for nn in range(output_size_row):
                  # filter_in_use_new = np.roll(myfilter, mm, axis=0)
                  # filter_in_use_new = np.roll(filter_in_use_new, nn, axis=1)
                  DLDZ_new = DLDZ[mm:mm+myfilter.shape[1], nn:nn+myfilter.shape[0]]
                  DLDA[ii][jj][kk][nn][mm] = np.sum(myfilter*DLDZ_new)

        dLdA = np.sum(DLDA,1) # TODO
        MM = dLdA

        for ii in range(dLdA.shape[0]):
          for jj in range(dLdA.shape[1]):
            dLdA[ii][jj] = np.transpose(MM[ii][jj])
        print('dLdA shape: ', dLdA.shape)

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels,
        kernel_size, weight_init_fn, bias_init_fn) # TODO # TODO
        self.downsample2d = Downsample2d(stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        # TODO
        # Call Conv1d_stride1
        # TODO
        AA = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(AA) # TODO

        # downsample
        # Z = None # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        # TODO
        DLDZ = self.downsample2d.backward(dLdZ) # TODO
        

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(DLDZ)

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(upsampling_factor) #TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels,
        kernel_size, weight_init_fn, bias_init_fn) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A) #TODO

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)  #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ) #TODO

        dLdA =  self.upsample1d.backward(delta_out) #TODO

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels,
        kernel_size, weight_init_fn, bias_init_fn) #TODO
        self.upsample2d = Upsample2d(upsampling_factor) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A) #TODO

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled)  #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ) #TODO

        dLdA = self.upsample2d.backward(delta_out) #TODO

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.A_shape = A.shape

        YY = A
        FLAT = np.zeros((A.shape[0], A.shape[1]*A.shape[2]))
        for ii in range(A.shape[0]):
          FLAT[ii] = YY[ii].flatten()
        Z = FLAT # TODO
        self.Z = Z

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        YY_back = np.zeros((self.A_shape))
        print('here: ', YY_back.shape)
        for ii in range(YY_back.shape[0]):
          YY_back[ii] = dLdZ[ii].reshape(self.A_shape[1],self.A_shape[2])

        dLdA = YY_back #TODO

        return dLdA #TODO


