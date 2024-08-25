import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        # print('test1')
        k = self.upsampling_factor
        # print('test2')
        total_length = np.shape(A)[-1]*k - (k-1)
        # print('test3')
        Z = np.array([[[0.0]*total_length for ii in range(np.shape(A)[1])] for jj in range(np.shape(A)[0])])
        # print('This is Z')
        # print(Z)
        # print('this is size of A:')
        # print(np.shape(A))
        for ii in range(len(A)):
          for jj in range(len(A[ii])):
           Z[ii][jj][0::k] = A[ii][jj]
        # Z[0::k] = A  # TODO
        # print('test5')
        # print('this is A:')
        # print(A)
        # print('this is Z')
        # print(Z)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        k = self.upsampling_factor
        new_length = (np.shape(dLdZ)[2]+(k-1))/k
        CC = np.array([[[0.0]*int(new_length) for ii in range(np.shape(dLdZ)[1])] for jj in range(np.shape(dLdZ)[0])])

        for ii in range(np.shape(dLdZ)[0]):
          for jj in range(np.shape(dLdZ)[1]):
            CC[ii][jj] = dLdZ[ii][jj][0::k]


        # print('Here')
        dLdA = CC  #TODO

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        k = self.downsampling_factor

        new_length = (np.shape(A)[2]+(k-1))/k
        # print('this is the new length: ', new_length)
        BB = np.array([[[0.0]*int(new_length) for ii in range(np.shape(A)[1])] for jj in range(np.shape(A)[0])])
        for ii in range(np.shape(A)[0]):
          for jj in range(np.shape(A)[1]):
            BB[ii][jj] = A[ii][jj][0::k]

        # print('this is the size of output of forward: ', np.shape(BB)[2])
        self.length_forward = np.shape(A)[2]

        Z = BB # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        k = self.downsampling_factor
        # print('-----------------')
        # print('this is the length from forward: ', self.length_forward)
        # print('this is the shape of the input: ', np.shape(dLdZ))
        # print('this is the downsampling factor: ',k)
        # print('this is the required output size according to the equation in 4.1: ', k* np.shape(dLdZ)[2]-(k-1))
        
        # if k == 5:
        #   total_length = 60
        # elif k == 6:
        #   total_length = 249
        # else:
        total_length = self.length_forward
        # total_length = np.shape(dLdZ)[-1]*k
        # if np.shape(dLdZ)[-1]%2 !=0 and 
        


        # print('this is the total length: ', total_length)
        # print('this is the shape of dLdZ (downsampling function): ', dLdZ.shape)
        
        Z = np.array([[[0.0]*total_length for ii in range(np.shape(dLdZ)[1])] for jj in range(np.shape(dLdZ)[0])])
        # print('this is the shape of Z: ', Z.shape)
        # print('this is k: ', k)
        for ii in range(np.shape(dLdZ)[0]):
          for jj in range(np.shape(dLdZ)[1]):
           Z[ii][jj][0::k] = dLdZ[ii][jj]
        
        # print('Shape of output: ', np.shape(Z))
        # print('-----------------')

        
        dLdA = Z  #TODO

        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        # print('Shape of input: ', np.shape(A))
        # print('Upsampling Factor: ', self.upsampling_factor)

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        k = self.upsampling_factor
        new_size_col = np.shape(A)[3]*k-(k-1)
        new_size_row = np.shape(A)[2]*k-(k-1)

        # print('new_size_col: ', new_size_col)
        # print('new_size_row: ', new_size_row)

        #first, work on rows
        BB = np.array([[[[0.0]*np.shape(A)[3] for ii in range(new_size_row)] for jj in range(np.shape(A)[1])] for kk in range(np.shape(A)[0])])

        # print('size BB: ', np.shape(BB))

        for ii in range(np.shape(BB)[0]):
          for jj in range(np.shape(BB)[1]):
            for kk in range(np.shape(A)[2]):
              # print('Here: ', A[ii][jj][kk])
              BB[ii][jj][kk*k] = A[ii][jj][kk]
        
        # print('This is BB:', BB)

        #second, work on columns
        CC = np.array([[[[0.0]*new_size_col for ii in range(new_size_row)] for jj in range(np.shape(A)[1])] for kk in range(np.shape(A)[0])])

        # print('size CC: ', np.shape(CC))

        for ii in range(np.shape(CC)[0]):
          for jj in range(np.shape(CC)[1]):
            for kk in range(np.shape(BB)[2]):
              CC[ii][jj][kk][0::k] = BB[ii][jj][kk]

        Z = CC # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        k = self.upsampling_factor
        CC = dLdZ
        new_size_col = int((np.shape(CC)[3]+(k-1))/k)
        new_size_row = int((np.shape(CC)[2]+(k-1))/k)
        WW = np.array([[[[0.0]*new_size_col for ii in range(new_size_row)] for jj in range(np.shape(CC)[1])] for kk in range(np.shape(CC)[0])])

        for ii in range(np.shape(WW)[0]):
          for jj in range(np.shape(WW)[1]):
            for kk in range(np.shape(WW)[2]):
              WW[ii][jj][kk] = CC[ii][jj][kk*k][0::k]

        dLdA = WW  #TODO

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        k = self.downsampling_factor
        new_size_row = int((np.shape(A)[2]-1)/k+1)
        new_size_col = int((np.shape(A)[3]-1)/k+1)

        BB = np.array([[[[0.0]*new_size_col for ii in range(new_size_row)] for jj in range(np.shape(A)[1])] for kk in range(np.shape(A)[0])])

        for ii in range(np.shape(BB)[0]):
          for jj in range(np.shape(BB)[1]):
            for kk in range(np.shape(BB)[2]):
              BB[ii][jj][kk] = A[ii][jj][kk*k][0::k]
        
        self.input_row = np.shape(A)[2]
        self.input_col = np.shape(A)[3]


        Z = BB # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        k = self.downsampling_factor

        row = self.input_row
        col = self.input_col

        new_size_col = col
        new_size_row = row
        
        # print('new_size_col: ', new_size_col)
        # print('new_size_row: ', new_size_row)

        #first, work on rows
        BB = np.array([[[[0.0]*np.shape(dLdZ)[3] for ii in range(new_size_row)] for jj in range(np.shape(dLdZ)[1])] for kk in range(np.shape(dLdZ)[0])])
        
        for ii in range(np.shape(BB)[0]):
          for jj in range(np.shape(BB)[1]):
            for kk in range(np.shape(dLdZ)[2]):
              # print('Here: ', A[ii][jj][kk])
              BB[ii][jj][kk*k] = dLdZ[ii][jj][kk]
        
        #second, work on columns
        CC = np.array([[[[0.0]*new_size_col for ii in range(new_size_row)] for jj in range(np.shape(BB)[1])] for kk in range(np.shape(BB)[0])])
        
        for ii in range(np.shape(CC)[0]):
          for jj in range(np.shape(CC)[1]):
            for kk in range(np.shape(BB)[2]):
              # print('Here: ', np.shape(CC))
              CC[ii][jj][kk][0::k] = BB[ii][jj][kk]

        dLdA = CC  #TODO

        return dLdA