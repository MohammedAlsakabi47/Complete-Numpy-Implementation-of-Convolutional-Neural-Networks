import numpy as np
from resampling import *
# from zmq.sugar import ZMQBindError

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        kernel_array = np.ones((self.kernel,self.kernel))
        output_size_row = A.shape[0] - self.kernel + 1
        output_size_col = A.shape[1] - self.kernel + 1
        input_width = A.shape[-2]


        in_channels = A.shape[1]
        out_channels = kernel_array.shape[0]
        kernel_size = kernel_array.shape[-1]
        output_size_col = A.shape[3]-kernel_size+1
        output_size_row = A.shape[2]-kernel_size+1
        self.out_row = output_size_col
        self.out_col = output_size_row
        batch_size = A.shape[0]

        # print('This is the batch size: ', batch_size)
        # print('out channels: ', out_channels)
        # print('in channels: ', in_channels)

        GG = np.zeros((batch_size, in_channels, output_size_row, output_size_col))

        self.IDX = np.zeros((batch_size, in_channels, output_size_row, output_size_col))
        self.A_shape = A.shape

        Pad_row = A.shape[2] - self.kernel
        Pad_col = A.shape[3] - self.kernel


        for ii in range(batch_size):
          for jj in range(in_channels):
            
              TT = A[ii][jj]
              RR = kernel_array

              # RR = np.pad(RR, ((0,Pad_row),(0,Pad_col)))

              for mm in range(output_size_col):
                for nn in range(output_size_row):
                  # print('mm: ', jj)
                  # filter_in_use_new = np.roll(RR, mm, axis=0)
                  # filter_in_use_new = np.roll(filter_in_use_new, nn, axis=1)
                  myarray = RR*TT[mm:mm+kernel_size , nn:nn+kernel_size]
                  # myarray = myarray[mm:(mm+self.kernel),nn:(nn+self.kernel)]
                  GG[ii][jj][mm][nn] = np.max(myarray)
                  # self.IDX[ii][kk][nn][mm] = np.argmax(filter_in_use_new*TT)
                  if len(np.where(TT.flatten() == GG[ii][jj][mm][nn])[0]) == 1:
                    self.IDX[ii][jj][mm][nn] = np.where(TT.flatten() == GG[ii][jj][mm][nn])[0][0]
                  
                  else:
                    temp = np.empty(TT.shape)
                    temp[:] = np.nan
                    temp[mm:(mm+self.kernel),nn:(nn+self.kernel)] = myarray[mm:(mm+self.kernel),nn:(nn+self.kernel)]
                    temp_idx_2D = np.where(temp == GG[ii][jj][mm][nn])
                    temp_idx_row = temp_idx_2D[0][0] + mm
                    temp_idx_col = temp_idx_2D[1][0] + nn
                    temp_idx = mm*input_width + nn
                    self.IDX[ii][jj][mm][nn] = temp_idx



        Z = GG

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        DLDA = np.zeros_like(self.A)
        in_row = self.A_shape[-2]
        in_col = self.A_shape[-1]

        PP = np.zeros((self.A.shape[0], self.A.shape[1], in_row, in_col))

        for ii in range(self.A.shape[0]):
          for jj in range(self.A.shape[1]):
            temp = PP[ii][jj].flatten()

            for mm in range(self.out_row):
              for nn in range(self.out_col):
                temp_idx = self.IDX[ii][jj][mm][nn]
                temp[int(temp_idx)] += dLdZ[ii][jj][mm][nn]
                PP[ii][jj] = temp.reshape(in_row,in_col)
        dLdA = PP

        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        kernel_array = np.ones((self.kernel,self.kernel))
        output_size_row = A.shape[0] - self.kernel + 1
        output_size_col = A.shape[1] - self.kernel + 1
        input_width = A.shape[-2]


        in_channels = A.shape[1]
        out_channels = kernel_array.shape[0]
        kernel_size = kernel_array.shape[-1]
        output_size_col = A.shape[3]-kernel_size+1
        output_size_row = A.shape[2]-kernel_size+1
        self.out_row = output_size_col
        self.out_col = output_size_row
        batch_size = A.shape[0]


        GG = np.zeros((batch_size, in_channels, output_size_row, output_size_col))


        # self.IDX = np.zeros((batch_size, in_channels, output_size_row, output_size_col))
        self.A_shape = A.shape

        Pad_row = A.shape[2] - self.kernel
        Pad_col = A.shape[3] - self.kernel

        for ii in range(batch_size):
          for jj in range(in_channels):
            
              TT = A[ii][jj]
              RR = kernel_array

              # RR = np.pad(RR, ((0,Pad_row),(0,Pad_col)))

              for mm in range(output_size_col):
                for nn in range(output_size_row):
                  # print('mm: ', jj)
                  # filter_in_use_new = np.roll(RR, mm, axis=0)
                  # filter_in_use_new = np.roll(filter_in_use_new, nn, axis=1)
                  myarray = RR*TT[mm:mm+kernel_size , nn:nn+kernel_size]
                  # myarray = myarray[mm:(mm+self.kernel),nn:(nn+self.kernel)]
                  GG[ii][jj][mm][nn] = np.mean(myarray)
                  # self.IDX[ii][kk][nn][mm] = np.argmax(filter_in_use_new*TT)
                  # if len(np.where(TT.flatten() == GG[ii][jj][mm][nn])[0]) == 1:
                    # self.IDX[ii][jj][mm][nn] = np.where(TT.flatten() == GG[ii][jj][mm][nn])[0][0]
                  
                  # else:
                  #   temp = np.empty(TT.shape)
                  #   temp[:] = np.nan
                  #   temp[mm:(mm+self.kernel),nn:(nn+self.kernel)] = myarray[mm:(mm+self.kernel),nn:(nn+self.kernel)]
                  #   temp_idx_2D = np.where(temp == GG[ii][jj][mm][nn])
                  #   temp_idx_row = temp_idx_2D[0][0] + mm
                  #   temp_idx_col = temp_idx_2D[1][0] + nn
                  #   temp_idx = mm*input_width + nn
                  #   self.IDX[ii][jj][mm][nn] = temp_idx

        Z = GG

        return Z
      

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size = self.A.shape[0]
        out_channels = dLdZ.shape[1]
        in_channels = self.A.shape[1]
        in_width = self.A.shape[2]
        in_hieght = self.A.shape[3]
        out_width = dLdZ.shape[-2]
        out_hieght = dLdZ.shape[-1]
        kernel_size = self.kernel

        # print('here: ', type(out_width*out_hieght))

        # DLDA = np.zeros((batch_size, in_channels, in_width*in_hieght, in_width, in_hieght))
        DLDA1 = np.zeros((batch_size, in_channels, in_width, in_hieght))
        DD = np.zeros((in_width, in_hieght))
        print("DD shape: ", DD.shape)
        for ii in range(batch_size):
          for jj in range(in_channels):
            for mm in range(out_hieght):
              for nn in range(out_width):
                
                DD[mm:mm+kernel_size, nn:nn+kernel_size] = (1/kernel_size**2)*dLdZ[ii][jj][mm][nn]

                # DLDA[ii][jj][mm+out_width*nn][mm:mm+kernel_size, nn:nn+kernel_size] = (1/kernel_size**2)*dLdZ[ii][jj][mm][nn]
                DLDA1[ii][jj][mm:mm+kernel_size, nn:nn+kernel_size] = DLDA1[ii][jj][mm:mm+kernel_size, nn:nn+kernel_size] + DD[mm:mm+kernel_size, nn:nn+kernel_size]


        # print("Here1: ", DLDA.shape)
        # print("Here2: ", np.sum(DLDA,2).shape)
        print("Here3: ", DLDA1.shape)
        # return np.sum(DLDA,2)
        return DLDA1

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 =  MaxPool2d_stride1(kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        AA = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(AA)
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        DD = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(DD)

        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """


        AA = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(AA)
        
        return Z
        


        # raise NotImplementedError

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        DD = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(DD)

        return dLdA


        # raise NotImplementedError
