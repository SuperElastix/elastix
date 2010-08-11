/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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

    /** Clear the container. */
    sampleContainer->Initialize();

    /** Set up a region iterator within the user specified image region. */
    typedef ImageRegionConstIteratorWithIndex<InputImageType> InputImageIterator;
    InputImageIterator iter( inputImage, this->GetCroppedInputImageRegion() );

    /** Take into account the possibility of a smaller bounding box around the mask */
    this->SetNumberOfSamples( this->m_RequestedNumberOfSamples );

    /** Determine the grid. */
    SampleGridIndexType index;
    SampleGridSizeType sampleGridSize;
    SampleGridIndexType sampleGridIndex =
      this->GetCroppedInputImageRegion().GetIndex();
    const InputImageSizeType & inputImageSize =
      this->GetCroppedInputImageRegion().GetSize();
    unsigned long numberOfSamplesOnGrid = 1;
    for (unsigned int dim = 0; dim < InputImageDimension; dim++)
    {
      /** The number of sample point along one dimension. */
      sampleGridSize[dim] = 1 +
        (( inputImageSize[dim] - 1 ) / this->GetSampleGridSpacing()[dim]);

      /** The position of the first sample along this dimension is
       * chosen to center the grid nicely on the input image region.
       */
      sampleGridIndex[dim] += (   inputImageSize[dim] -
        ( (sampleGridSize[dim] - 1) * this->GetSampleGridSpacing()[dim] +1 )   ) / 2;

      /** Update the number of samples on the grid. */
      numberOfSamplesOnGrid *= sampleGridSize[dim];
    }

    /** Prepare for looping over the grid. */
    unsigned int dim_z = 1;
    unsigned int dim_t = 1;
    if ( InputImageDimension > 2 )
    {
      dim_z = sampleGridSize[ 2 ];
      if ( InputImageDimension > 3 )
      {
        dim_t = sampleGridSize[ 3 ];
      }
    }
    index = sampleGridIndex;

    if ( mask.IsNull() )
    {
      /** Ugly loop over the grid. */
      for ( unsigned int t = 0; t < dim_t; t++)
      {
        for ( unsigned int z = 0; z < dim_z; z++)
        {
          for ( unsigned int y = 0; y < sampleGridSize[1]; y++)
          {
            for ( unsigned int x = 0; x < sampleGridSize[0]; x++)
            {
              ImageSampleType tempsample;

              // Get sampled fixed image value.
              tempsample.m_ImageValue = inputImage->GetPixel(index);

              // Translate index to point.
              inputImage->TransformIndexToPhysicalPoint(
                index, tempsample.m_ImageCoordinates );

              // Jump to next position on grid.
              index[ 0 ] += this->m_SampleGridSpacing[ 0 ];

              // Store sample in container.
              sampleContainer->push_back( tempsample );

            } // end x
            index[ 0 ] = sampleGridIndex[0];
            index[ 1 ] += this->m_SampleGridSpacing[ 1 ];

          } // end y
          if ( InputImageDimension > 2 )
          {
            index[ 1 ] = sampleGridIndex[ 1 ];
            index[ 2 ] += this->m_SampleGridSpacing[ 2 ];
          }
        } // end z
        if ( InputImageDimension > 3 )
        {
          index[ 2 ] = sampleGridIndex[ 2 ];
          index[ 3 ] += this->m_SampleGridSpacing[ 3 ];
        }
      } // end t

    } // end if no mask
    else
    {
      if ( mask->GetSource() )
      {
        mask->GetSource()->Update();
      }
      /* Ugly loop over the grid; checks also if a sample falls within the mask. */
      for ( unsigned int t = 0; t < dim_t; t++)
      {
        for ( unsigned int z = 0; z < dim_z; z++)
        {
          for ( unsigned int y = 0; y < sampleGridSize[1]; y++)
          {
            for ( unsigned int x = 0; x < sampleGridSize[0]; x++)
            {
              ImageSampleType tempsample;

              // Translate index to point.
              inputImage->TransformIndexToPhysicalPoint(
                index, tempsample.m_ImageCoordinates );

              if (  mask->IsInside( tempsample.m_ImageCoordinates )  )
              {
                // Get sampled fixed image value.
                tempsample.m_ImageValue = inputImage->GetPixel( index );

                // Store sample in container.
                sampleContainer->push_back(tempsample);

              } // end if in mask
              // Jump to next position on grid
              index[ 0 ] += this->m_SampleGridSpacing[ 0 ];

            } // end x
            index[ 0 ] = sampleGridIndex[ 0 ];
            index[ 1 ] += this->m_SampleGridSpacing[ 1 ];

          } // end y
          if (InputImageDimension > 2)
          {
            index[ 1 ] = sampleGridIndex[ 1 ];
            index[ 2 ] += this->m_SampleGridSpacing[ 2 ];
          }
        } // end z
        if ( InputImageDimension > 3 )
        {
          index[ 2 ] = sampleGridIndex[ 2 ];
          index[ 3 ] += this->m_SampleGridSpacing[ 3 ];
        }
      } // end t
    } // else (if mask exists)

  } // end GenerateData


  /**
   * ******************* SetNumberOfSamples *******************
   */

  template< class TInputImage >
    void
    ImageGridSampler< TInputImage >
    ::SetNumberOfSamples( unsigned long nrofsamples )
  {
    /** Store what the user wanted */
    if ( this->m_RequestedNumberOfSamples != nrofsamples )
    {
      this->m_RequestedNumberOfSamples = nrofsamples;
      this->Modified();
    }

    /** Compute an isotropic grid spacing which realises the nrofsamples
     * approximately */
    if ( nrofsamples != 0 )
    {
      /** Compute the grid spacing needed to achieve the NumberOfSamplesForExactGradient. */
      const unsigned long allvoxels = this->GetInputImageRegion().GetNumberOfPixels();
      const double allvoxelsd = static_cast<double>( allvoxels );
      const double nrofsamplesd = static_cast<double>( nrofsamples );
      const double indimd = static_cast<double>( InputImageDimension );

      /** compute isotropic gridspacing */
      const double fraction = allvoxelsd / nrofsamplesd;
      int gridspacing = static_cast<int>(
        vnl_math_rnd( vcl_pow( fraction, 1.0/indimd ) )   );
      gridspacing = vnl_math_max( 1, gridspacing );

      /** Set gridspacings for all dimensions
       * Do not use the SetSampleGridSpacing function because it calls
       * SetNumberOfSamples(0) internally. */
      SampleGridSpacingType gridspacings;
      gridspacings.Fill( gridspacing );
      if ( this->GetSampleGridSpacing() != gridspacings )
      {
        this->m_SampleGridSpacing = gridspacings;
        this->Modified();
      }
    }

  } // end SetNumberOfSamples


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

