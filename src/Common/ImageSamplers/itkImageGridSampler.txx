#ifndef __ImageGridSampler_txx
#define __ImageGridSampler_txx

#include "itkImageGridSampler.h"

#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{


  /**
	 * ******************* GenerateData *******************
	 */
  
  template< class TInputImage >
    void
    ImageGridSampler< TInputImage >
    ::GenerateData( void )
  {
    /** Get handles to the input image, output sample container, and the mask. */
    InputImageConstPointer inputImage = this->GetInput();
    typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();
    typename MaskType::ConstPointer mask = this->GetMask();


		/** Set up a region interator within the user specified image region.*/
	  typedef ImageRegionConstIteratorWithIndex<InputImageType> InputImageIterator;
		InputImageIterator iter( inputImage, this->GetInputImageRegion() );

    /** Determine the grid */ 
    SampleGridSizeType sampleGridSize;
		SampleGridIndexType sampleGridIndex;
		SampleGridIndexType index;
		const InputImageSizeType & inputImageSize =
			this->GetInputImageRegion().GetSize();
		unsigned long numberOfSamplesOnGrid = 1;
		for (unsigned int dim = 0; dim < InputImageDimension; dim++)
		{
			/** the number of sample point along one dimension */
			sampleGridSize[dim] = 1 + 
				(( inputImageSize[dim] - 1 ) / this->GetSampleGridSpacing()[dim]);

			/** the position of the first sample along this dimension is 
			 * chosen to center the grid nicely on the image */
			sampleGridIndex[dim] = (   inputImageSize[dim] - 
				( (sampleGridSize[dim] - 1) * this->GetSampleGridSpacing()[dim] +1 )   ) / 2;

      /** Update the number of samples on the grid */
			numberOfSamplesOnGrid *= sampleGridSize[dim];
		}

    /** Prepare for looping over the grid */
		unsigned int dim_z = 1;
		if (InputImageDimension > 2)
		{
			dim_z = sampleGridSize[2];
		}
		index = sampleGridIndex;

		if ( mask.IsNull() )
		{
			/** ugly loop over the grid */
			for ( unsigned int z = 0; z < dim_z; z++)
			{
				for ( unsigned int y = 0; y < sampleGridSize[1]; y++)
				{
					for ( unsigned int x = 0; x < sampleGridSize[0]; x++)
					{
            ImageSampleType tempsample;
						// Get sampled fixed image value
						tempsample.m_ImageValue = inputImage->GetPixel(index);
						// Translate index to point
						inputImage->TransformIndexToPhysicalPoint(
							index, tempsample.m_ImageCoordinates );
			 			// Jump to next position on grid
						index[0] += this->m_SampleGridSpacing[0];
						// store sample in container
						sampleContainer->push_back(tempsample);
					} // end x
					index[0] = sampleGridIndex[0];
					index[1] += this->m_SampleGridSpacing[1];
				} // end y
				if ( InputImageDimension > 2)
				{
					index[1] = sampleGridIndex[1];
				  index[2] += this->m_SampleGridSpacing[2];
				}
		  } // end z
			
		} // end if no mask
		else
		{
      if ( mask->GetSource() )
      {
        mask->GetSource()->Update();
      }
		  /* Ugly loop over the grid; checks also if a sample falls within the mask */
			for ( unsigned int z = 0; z < dim_z; z++)
			{
				for ( unsigned int y = 0; y < sampleGridSize[1]; y++)
				{
					for ( unsigned int x = 0; x < sampleGridSize[0]; x++)
					{
            ImageSampleType tempsample;
						// Translate index to point
						inputImage->TransformIndexToPhysicalPoint(
							index, tempsample.m_ImageCoordinates );
			 			if (  mask->IsInside( tempsample.m_ImageCoordinates )  )
						{
							// Get sampled fixed image value
							tempsample.m_ImageValue = inputImage->GetPixel(index);
							// store sample in container
              sampleContainer->push_back(tempsample);			
						} // end if in mask
						// Jump to next position on grid
						index[0] += this->m_SampleGridSpacing[0];
					} // end x
					index[0] = sampleGridIndex[0];
					index[1] += this->m_SampleGridSpacing[1];
				} // end y
				if (InputImageDimension > 2)
				{
					index[1] = sampleGridIndex[1];
					index[2] += this->m_SampleGridSpacing[2];
				}
		  } // end z
					
		} // else (if mask exists)
   
  } // end GenerateData


  /**
	 * ******************* PrintSelf *******************
	 */
  
  template< class TInputImage >
    void
    ImageGridSampler< TInputImage >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );
  } // end PrintSelf



} // end namespace itk

#endif // end #ifndef __ImageGridSampler_txx

