/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkComputeImageExtremaFilter_hxx
#define itkComputeImageExtremaFilter_hxx
#include "itkComputeImageExtremaFilter.h"

namespace itk
{

/**
* ********************* Constructor ****************************
*/

template< typename TInputImage >
  ComputeImageExtremaFilter< TInputImage >
  ::ComputeImageExtremaFilter()
{
  this->m_UseMask = false;
  this->m_SameGeometry = false;
}

template< typename TInputImage >
void
ComputeImageExtremaFilter< TInputImage >
::BeforeThreadedGenerateData()
{
  if( !this->m_UseMask )
  {
    Superclass::BeforeThreadedGenerateData();
  }
  else
  {
    ThreadIdType numberOfThreads = this->GetNumberOfThreads();

    // Resize the thread temporaries
    m_Count.SetSize( numberOfThreads );
    m_SumOfSquares.SetSize( numberOfThreads );
    m_ThreadSum.SetSize( numberOfThreads );
    m_ThreadMin.SetSize( numberOfThreads );
    m_ThreadMax.SetSize( numberOfThreads );

    // Initialize the temporaries
    m_Count.Fill( NumericTraits < SizeValueType >::ZeroValue() );
    m_ThreadSum.Fill( NumericTraits < RealType >::ZeroValue() );
    m_SumOfSquares.Fill( NumericTraits < RealType >::ZeroValue() );
    m_ThreadMin.Fill( NumericTraits < PixelType >::max() );
    m_ThreadMax.Fill( NumericTraits < PixelType >::NonpositiveMin() );
    //this->SameGeometry();

    if( this->GetImageSpatialMask() )
    {
      this->SameGeometry();
    }
    else
    {
      this->m_SameGeometry = false;
    }
  }
}

template< typename TInputImage >
void
ComputeImageExtremaFilter< TInputImage >
::SameGeometry()
{
  if( this->GetInput()->GetLargestPossibleRegion().GetSize() == this->m_ImageSpatialMask->GetImage()->GetLargestPossibleRegion().GetSize()
    && this->GetInput()->GetOrigin() == this->m_ImageSpatialMask->GetImage()->GetOrigin() )
  {
    this->m_SameGeometry = true;
  }
}

template< typename TInputImage >
void
ComputeImageExtremaFilter< TInputImage >
::AfterThreadedGenerateData()
{
  if( !this->m_UseMask )
  {
    Superclass::AfterThreadedGenerateData();
  }
  else
  {
    ThreadIdType    i;
    SizeValueType   count;
    RealType        sumOfSquares;

    ThreadIdType numberOfThreads = this->GetNumberOfThreads();

    PixelType minimum;
    PixelType maximum;
    RealType  mean;
    RealType  sigma;
    RealType  variance;
    RealType  sum;

    sum = sumOfSquares = NumericTraits< RealType >::ZeroValue();
    count = 0;

    // Find the min/max over all threads and accumulate count, sum and
    // sum of squares
    minimum = NumericTraits< PixelType >::max();
    maximum = NumericTraits< PixelType >::NonpositiveMin();
    for( i = 0; i < numberOfThreads; ++i )
    {
      count += m_Count[ i ];
      sum += m_ThreadSum[ i ];
      sumOfSquares += m_SumOfSquares[ i ];

      if( m_ThreadMin[ i ] < minimum )
      {
        minimum = m_ThreadMin[ i ];
      }
      if( m_ThreadMax[ i ] > maximum )
      {
        maximum = m_ThreadMax[ i ];
      }
    }
    m_Count.Fill( NumericTraits< SizeValueType >::ZeroValue() );
    m_SumOfSquares.Fill( NumericTraits< RealType >::ZeroValue() );
    m_ThreadSum.Fill( NumericTraits< RealType >::ZeroValue() );
    m_ThreadMin.Fill( NumericTraits< PixelType >::max() );
    m_ThreadMax.Fill( NumericTraits< PixelType >::NonpositiveMin() );
    // compute statistics
    mean = sum / static_cast< RealType >( count );

    // unbiased estimate
    variance = ( sumOfSquares - ( sum * sum / static_cast< RealType >( count )) )
      / (static_cast< RealType >( count ) - 1);
    sigma = std::sqrt( variance );

    // Set the outputs
    this->GetMinimumOutput()->Set( minimum );
    this->GetMaximumOutput()->Set( maximum );
    this->GetMeanOutput()->Set( mean );
    this->GetSigmaOutput()->Set( sigma );
    this->GetVarianceOutput()->Set( variance );
    this->GetSumOutput()->Set( sum );
  }
}

template< typename TInputImage >
void
ComputeImageExtremaFilter< TInputImage >
::ThreadedGenerateData( const RegionType & outputRegionForThread,
  ThreadIdType threadId )
{
  if( !this->m_UseMask )
  {
    Superclass::ThreadedGenerateData( outputRegionForThread, threadId );
  }
  else
  {
    if( this->GetImageSpatialMask() )
    {
      this->ThreadedGenerateDataImageSpatialMask( outputRegionForThread, threadId );
    }
    if( this->GetImageMask() )
    {
      this->ThreadedGenerateDataImageMask( outputRegionForThread, threadId );
    }
  }
} // end ThreadedGenerateData()

template< typename TInputImage >
void
ComputeImageExtremaFilter< TInputImage >
::ThreadedGenerateDataImageSpatialMask( const RegionType & outputRegionForThread,
    ThreadIdType threadId )
{
  const SizeValueType size0 = outputRegionForThread.GetSize( 0 );
  if( size0 == 0 )
  {
    return;
  }
  RealType  realValue;
  PixelType value;

  RealType sum = NumericTraits< RealType >::ZeroValue();
  RealType sumOfSquares = NumericTraits< RealType >::ZeroValue();
  SizeValueType count = NumericTraits< SizeValueType >::ZeroValue();
  PixelType min = NumericTraits< PixelType >::max();
  PixelType max = NumericTraits< PixelType >::NonpositiveMin();

  if( this->m_SameGeometry )
  {
    ImageRegionConstIterator< TInputImage > it (this->GetInput(), outputRegionForThread );
    for( it.GoToBegin(); !it.IsAtEnd(); ++it )
    {
      if( this->m_ImageSpatialMask->GetImage()->GetPixel( it.GetIndex()) != NumericTraits< PixelType >::ZeroValue() )
      {
        value = it.Get();
        realValue = static_cast<RealType>( value );

        min = vnl_math_min( min, value );
        max = vnl_math_max( max, value );

        sum += realValue;
        sumOfSquares += ( realValue * realValue );
        ++count;
      }
    } // end for
  }
  else
  {
    ImageRegionConstIterator< TInputImage > it( this->GetInput(), outputRegionForThread );
    for( it.GoToBegin(); !it.IsAtEnd(); ++it )
    {
      PointType point;
      this->GetInput()->TransformIndexToPhysicalPoint( it.GetIndex(), point );
      if( this->m_ImageSpatialMask->IsInside( point ) )
      {
        value = it.Get();
        realValue = static_cast<RealType>( value );

        min = vnl_math_min( min, value );
        max = vnl_math_max( max, value );

        sum += realValue;
        sumOfSquares += ( realValue * realValue );
        ++count;
      }
    } // end for
  } // end if

  m_ThreadSum[ threadId ] = sum;
  m_SumOfSquares[ threadId ] = sumOfSquares;
  m_Count[ threadId ] = count;
  m_ThreadMin[ threadId ] = min;
  m_ThreadMax[ threadId ] = max;
} // end ThreadedGenerateDataImageSpatialMask()


template< typename TInputImage >
void
ComputeImageExtremaFilter< TInputImage >
::ThreadedGenerateDataImageMask( const RegionType & outputRegionForThread,
  ThreadIdType threadId )
{
  const SizeValueType size0 = outputRegionForThread.GetSize( 0 );
  if( size0 == 0 )
  {
    return;
  }
  RealType  realValue;
  PixelType value;

  RealType sum = NumericTraits< RealType >::ZeroValue();
  RealType sumOfSquares = NumericTraits< RealType >::ZeroValue();
  SizeValueType count = NumericTraits< SizeValueType >::ZeroValue();
  PixelType min = NumericTraits< PixelType >::max();
  PixelType max = NumericTraits< PixelType >::NonpositiveMin();

  ImageRegionConstIterator< TInputImage > it ( this->GetInput(), outputRegionForThread );
  it.GoToBegin();

  // do the work
  while( !it.IsAtEnd() )
  {
    PointType point;
    this->GetInput()->TransformIndexToPhysicalPoint( it.GetIndex(), point );
    if( this->m_ImageMask->IsInside( point ) )
    {
      value = it.Get();
      realValue = static_cast<RealType>( value );

      min = vnl_math_min( min, value );
      max = vnl_math_max( max, value );

      sum += realValue;
      sumOfSquares +=  ( realValue * realValue );
      ++count;
    }
    ++it;
  }// end while

  m_ThreadSum[ threadId ] = sum;
  m_SumOfSquares[ threadId ] = sumOfSquares;
  m_Count[ threadId ] = count;
  m_ThreadMin[ threadId ] = min;
  m_ThreadMax[ threadId ] = max;
} // end ThreadedGenerateDataImageMask()

} // end namespace itk
#endif
