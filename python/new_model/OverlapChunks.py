import numpy as np
class OverlapChunks( object ):
    """
    Split array into overlapping windows and re-join later
    
    Arrays that are 'too small' to chunk evenly are expanded
    using numpy.pad to the correct size.
    
    """
    def __init__(self, array, window, overlap, inplace=False, **kwargs):
        """
        Initialize the splitter
        
        Arguments:
            array (numpy.ndarray) : A n-dimensional array to split
            window (int, array-like) : An integer specifying the window size
                in all dimensions OR an array-like object specifying window
                size for each dimension.
                Note that this value should be more than twice the size of the
                overlap.
            overlap (int, array-like) : An integer specifying by how many
                points the windows should overlap in all dimensions OR an 
                array-like object specifying overlap for each dimension.
                Note that this is the overlap for each side, so for an overlap
                of 4, that is a total overlap of 8; 4 for left, 4 for right, 
                4 for top, etc.
        Keyword arguments:
            inplace (bool) : By default (False), a copy of the input array is
                used for chunking, with data rejoined into the copied array.
                If True, then no copy is taken and the ORIGINAL array input
                is overwritten during joining. This will save some memory, but
                all input data will be lost if any rejoined chunks have been
                modified.
            **kwargs : Silently ignores other keywords
            
        """
        # Ensure inputs are list like for later iteration
        if isinstance(window,  int): window = [window]
        if isinstance(overlap, int): overlap = [overlap]
        # Convert to numpy arrays
        window  = np.asarray( window )
        overlap = np.asarray( overlap )
        # If sizes do NOT match input array, repeat to match
        if window.size  != array.ndim: window  = np.repeat( window,  array.ndim )
        if overlap.size != array.ndim: overlap = np.repeat( overlap, array.ndim )
        # Determine padding of input array, width of window withOUT padding, and
        # how to subset chunks to remove overlap when joining
        padding = []
        widths  = []
        subset  = []
        for i, nn in enumerate( array.shape ):
            if overlap[i] == 0:
                padding.append( [0,0] )
            else:
                if window[i] <= (2*overlap[i]):
                    raise Exception( "Overlap size is larger than window!!!" )
                    
                nWin = (nn+overlap[i]) / window[i] # Determine number of windows across dimension with left pad included
                # Determine left and right padding for the dimension
                padding.append(
                    [overlap[i], int((np.ceil(nWin)-nWin)*window[i]) + 1]
                )
            # Set subset used for rejoining chunks
            subset.append(
                slice(overlap[i], window[i]-overlap[i])
            )
            # Set width of data with OUT overlap
            widths.append( window[i]-2*overlap[i] )
            
        self._array  = array if inplace else array.copy()
        self.window  = window
        self.overlap = overlap
        self.padding = padding
        self.widths  = widths
        self.subset  = tuple(subset)
        
        self.nChunks = None
        self.chunk   = None
    def __iter__(self):
      yield from self.split()
    @property
    def array(self):
        return self._array
    
    def split(self, mode='reflect'):
        """
        Generator that yields chunks
        Keyword arguments:
            mode (str) : Mode used in numpy.pad to ensure input array is
                'correct' size for chunking
        """
        tmp = np.lib.stride_tricks.sliding_window_view(
            np.pad( self._array, self.padding, mode=mode ),
            self.window
        )[ tuple( [slice(None,None,width) for width in self.widths] ) ]
        # Set chunk number and # chunks in each dimension for joining
        self.chunk   = 0
        self.nChunks = tmp.shape[:self._array.ndim]
#         print(tmp.shape)
#         print(np.reshape(tmp, ( np.product(self.nChunks), *tmp.shape[self._array.ndim:])).shape )
        # Yield the chunks from the sliding window view
        return(np.reshape(tmp, ( np.product(self.nChunks), *tmp.shape[self._array.ndim:])))
#         yield from np.reshape( 
#             tmp, 
#             ( np.product(self.nChunks), *tmp.shape[self._array.ndim:])
#         )
    def join(self, chunk):
        """
        Join chunks back together
        
        This method can be called after each chunk is processed OR
        called on a list of chunks.
        
        Note:
            Chunks MUST be passed back to this method in the order they were
            produced.
        
        Arguments:
            chunk (numpy.ndarray, array-like) : A single chunk OR
                and array-like object containing chunks to join back into
                a single array.
        Example:
        
            >>> chunks = OverlapChunks( zz, 32, 8 )
            >>> for chunk in chunks.split():
            >>>     chunks.join( processChunk( chunk ) )
            >>> res = chunks.array
            
        """
        # If array-like input, iterate over passing values into this method
        if isinstance(chunk, (list,tuple)):
            for c in chunk:
                self.join( c )
        
        subset = list( self.subset )
        idx    = np.unravel_index( self.chunk, self.nChunks )
        slices = []
        for i in range(self._array.ndim):
            start = idx[i] * self.widths[i]
            end   = start + self.widths[i]
            # If the end of the chunk goes past original array,
            # Force 'end' to be size of array and update the
            # subset slice to be correct size
            if end > self._array.shape[i]:
                end = self._array.shape[i]
                subset[i] = slice( subset[i].start, subset[i].start+(end-start), subset[i].step )
            slices.append( slice( start, end ) )
#        print(tuple(subset))
        self._array[ tuple(slices) ] = chunk[ tuple(subset) ]
        self.chunk += 1
        
def test(nn = 100, ndims = 2, verbose = False):
    nFail = 0
    for i in range(nn):
        overlap = [np.random.randint( 2,  15 ) for i in range(ndims)]
        window  = [np.random.randint( o*2+1, o*5) for o in overlap]
        zz      = np.random.random( [w*np.random.randint(10, 15) for w in window] )
        chunks  = OverlapChunks( zz, window, overlap )
        for chunk in chunks.split():
            chunks.join( chunk )
        nFail += ( (zz != chunks.array).sum() != 0)
        if verbose:
            print( f"Overlap : {overlap}" )
            print( f"Window  : {window}" )
            print( f"Shape   : {zz.shape}" )
            print( f"Unequal : {(zz != chunks.array).sum()}" )
    print( f"{nFail/nn}% Failed" )
    
    
'''
import numpy as np
import sys
sys.path.insert(1, '../')
from new_model.OverlapChunks import OverlapChunks, test
test(nn=1)
'''
    