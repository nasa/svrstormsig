import numpy as np

class DataScaler():

  def __init__(self, nbytes = 2, signed=True):
    """
    Initialize scaler for given byte-depth and signed data type

    Arguments:
      nbytes (int) : Number of bytes to scale data to
      signed (bool) : If set, signed integers are used, if False, unsigned

    """

    self._nbytes = None
    self._dfmt   = None
    self._dtype  = None
    self._oMin   = None
    self._oMax   = None
    self._miss   = None
    self._signed = signed
    self.nbytes  = nbytes

  @property
  def nbytes(self):
    return self._nbytes
  @nbytes.setter
  def nbytes(self, val):
    """
    Updates various information used for scaling when byte-depth is changed

    When the nbytes attribute is changed, the min/max of the packed data
    must be updated, along with the numpy data type to use. All that
    is handled in this setter method

    """
    self._nbytes = val                                                          # User input value
    self._dfmt   = f'i{val}' if self.signed else f'u{val}'                      # Generate type string
    self._dtype  = np.dtype( self._dfmt )                                       # Generate numpy dtype based on string
    info         = np.iinfo( self._dtype )                                      # Get information about the numpy data type
    self._miss   = info.min                                                     # Set missing value to minimum value that can be represented by given bit depth
    self._oMin   = self._miss + 1                                               # Set the output minimum value to one (1) greater than the missing value
    self._oMax   = info.max                                                     # Set the output maximum to the maximum value that can be represented by given bit depth

  @property
  def signed(self):
    return self._signed
  @signed.setter
  def signed(self, val):
    """
    Updates signed attribute along with range of output values

    When the signed flag changes, the nbytes attribute is reset so
    that the nbytes.setter method is run to update the missing value
    and output minimum/maximum values

    """

    if not isinstance(val, bool):
      raise Exception( 'Must be of type bool!' )
    self._signed = val
    self.nbytes  = self._nbytes 

  @property
  def missing_value(self):
    """Read only value for missing_value; set by nbytes.setter"""

    return self._miss

  @property
  def _FillValue(self):
    """Read only value for _FillValue; set by nbytes.setter"""

    return self._miss

  @property
  def dtype(self):
    """Read only value for data type; set by nbytes.setter"""

    return self._dtype

  def computeScale(self, dataMin, dataMax):
    """
    Compute scale factor based on data minimum/maximum
    
    Arguments:
      dataMin (int,float) : Minimum value of data to scale
      dataMax (int,float) : Maximum value of data to scale
    
    
    Returns:
      int,float : Scale factor for packing data

    """

    if dataMax == dataMin:                                                      # If the min and max are the same, scale factor is 1
      return 1
    return (dataMax - dataMin) / (self._oMax - self._oMin)

  def computeOffset(self, dataMin, scale):
    """
    Compute add offset based on data minimum and scale factor
    
    Arguments:
      dataMin (int,float) : Minimum value of data to scale
      scale (int,float) : Scale factor for packing data; computed by computeScale() 
    
    Returns:
      int,float : Add offset for packing data

    """

    return -(self._oMin*scale - dataMin)

  def packData(self, data, scale, offset):
    """
    Pack the data in the specified number of bytes
    
    Arguments:
      data (numpy.ndarray, numpy.ma.MaskedArray) : Data to pack
      scale (int,float) : Scale factor for packing data; computed by computeScale() 
      offset (int,float) : Add offset for packing data

    Returns:
      numpy.ndarray : Packed data

    """

    if isinstance( data, np.ma.core.MaskedArray ):
      index = data.mask                                                         # Get mask
    else:
      index = ~np.isfinite( data )                                              # Locate all NaN (i.e., missing values)
    data = np.round( (data - offset) / scale)
    data = np.clip( data, self._oMin, self._oMax ).astype( self._dtype )        # Clip to range to ensure nothing out-of-bounds and force type
    data[index] = self._miss

    #if any(index):
    #  print('Runtime warning is of no concern, NaN/Inf values in data are set to missing values.')

    return data

  def unpackData(self, data, scale, offset):
    """
    Unpack the data into usable values
    
    Arguments:
      data (numpy.ndarray, numpy.ma.MaskedArray) : Packed data values to unpack
      scale (int,float) : Scale factor for unpacking data; computed by computeScale() 
      offset (int,float) : Add offset for unpacking data

    Returns:
      numpy.ndarray : Unpacked data

    """

    index = data == self._miss
    data  = data * scale + offset
    data[index] = np.nan
    return data
 
  def scaleData( self, data ):
    """
    Scale data to integer type

    Will scale the input data to a n byte integer. Smallest value
    of integer type is reserved for missing data.

    Arguments:
      data (np.ndarray) : Numpy array of data to scale

    Keyword arguments:
      nbytes (int) : Number of bytes to scale to

    Returns:
      tuple : Scaled data, scaling factor, add offset

    """

    dtype   = data.dtype
    dataMin = np.nanmin(data)                                                   # Compute minimum of data
    dataMax = np.nanmax(data)                                                   # Compute maximum of data
    scale   = self.computeScale( dataMin, dataMax )                             # Compute scale factor
    offset  = self.computeOffset( dataMin, scale )                              # Compute add offset
    data    = self.packData( data, scale, offset )                              # Pack the data 

    return data, dtype.type(scale), dtype.type(offset)                          # Return the scaled data, scale factor, add offset, and missing value
