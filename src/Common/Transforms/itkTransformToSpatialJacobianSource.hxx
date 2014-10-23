/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkTransformToSpatialJacobianSource_hxx
#define __itkTransformToSpatialJacobianSource_hxx

#include "itkTransformToSpatialJacobianSource.h"

#include "itkAdvancedIdentityTransform.h"
#include "itkProgressReporter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vnl/vnl_copy.h"

namespace itk
{

/**
 * Constructor
 */
template< class TOutputImage, class TTransformPrecisionType >
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::TransformToSpatialJacobianSource()
{
  this->m_OutputSpacing.Fill( 1.0 );
  this->m_OutputOrigin.Fill( 0.0 );
  this->m_OutputDirection.SetIdentity();

  SizeType size;
  size.Fill( 0 );
  this->m_OutputRegion.SetSize( size );

  IndexType index;
  index.Fill( 0 );
  this->m_OutputRegion.SetIndex( index );

  this->m_Transform = AdvancedIdentityTransform< TTransformPrecisionType, ImageDimension >::New();

  // Check if the output pixel type is valid
  // TODO: should maybe be done at compile time, using concept checking.
  const unsigned int pixrow  = PixelType::RowDimensions;
  const unsigned int pixcol  = PixelType::ColumnDimensions;
  const unsigned int spatrow = SpatialJacobianType::RowDimensions;
  const unsigned int spatcol = SpatialJacobianType::ColumnDimensions;
  if( ( pixrow != spatrow ) || ( pixcol != spatcol ) )
  {
    itkExceptionMacro( "The specified output image type is not allowed for this filter" );
  }

} // end Constructor


/**
 * Print out a description of self
 *
 * \todo Add details about this class
 */
template< class TOutputImage, class TTransformPrecisionType >
void
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "OutputRegion: " << this->m_OutputRegion << std::endl;
  os << indent << "OutputSpacing: " << this->m_OutputSpacing << std::endl;
  os << indent << "OutputOrigin: " << this->m_OutputOrigin << std::endl;
  os << indent << "OutputDirection: " << this->m_OutputDirection << std::endl;
  os << indent << "Transform: " << this->m_Transform.GetPointer() << std::endl;

} // end PrintSelf()


/**
 * Set the output image size.
 */
template< class TOutputImage, class TTransformPrecisionType >
void
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::SetOutputSize( const SizeType & size )
{
  this->m_OutputRegion.SetSize( size );
}


/**
 * Get the output image size.
 */
template< class TOutputImage, class TTransformPrecisionType >
const typename TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::SizeType
& TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::GetOutputSize()
{
  return this->m_OutputRegion.GetSize();
}

/**
 * Set the output image index.
 */
template< class TOutputImage, class TTransformPrecisionType >
void
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::SetOutputIndex( const IndexType & index )
{
  this->m_OutputRegion.SetIndex( index );
}


/**
 * Get the output image index.
 */
template< class TOutputImage, class TTransformPrecisionType >
const typename TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::IndexType
& TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::GetOutputIndex()
{
  return this->m_OutputRegion.GetIndex();
}

/**
 * Set the output image spacing.
 */
template< class TOutputImage, class TTransformPrecisionType >
void
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::SetOutputSpacing( const double * spacing )
{
  SpacingType s( spacing );
  this->SetOutputSpacing( s );

} // end SetOutputSpacing()


/**
 * Set the output image origin.
 */
template< class TOutputImage, class TTransformPrecisionType >
void
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::SetOutputOrigin( const double * origin )
{
  OriginType p( origin );
  this->SetOutputOrigin( p );

}


/** Helper method to set the output parameters based on this image */
template< class TOutputImage, class TTransformPrecisionType >
void
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::SetOutputParametersFromImage( const ImageBaseType * image )
{
  if( !image )
  {
    itkExceptionMacro( << "Cannot use a null image reference" );
  }

  this->SetOutputOrigin( image->GetOrigin() );
  this->SetOutputSpacing( image->GetSpacing() );
  this->SetOutputDirection( image->GetDirection() );
  this->SetOutputRegion( image->GetLargestPossibleRegion() );

} // end SetOutputParametersFromImage()


/**
 * Set up state of filter before multi-threading.
 * InterpolatorType::SetInputImage is not thread-safe and hence
 * has to be set up before ThreadedGenerateData
 */
template< class TOutputImage, class TTransformPrecisionType >
void
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::BeforeThreadedGenerateData( void )
{
  if( !this->m_Transform )
  {
    itkExceptionMacro( << "Transform not set" );
  }

  // Check whether we can use a fast path for resampling. Fast path
  // can be used if the transformation is linear. Transform respond
  // to the IsLinear() call.
  if( this->m_Transform->IsLinear() )
  {
    this->LinearGenerateData();
  }

} // end BeforeThreadedGenerateData()


/**
 * ThreadedGenerateData
 */
template< class TOutputImage, class TTransformPrecisionType >
void
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::ThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread,
  ThreadIdType threadId )
{
  // In case of linear transforms, the computation has already been
  // completed in the BeforeThreadedGenerateData
  if( this->m_Transform->IsLinear() )
  {
    return;
  }

  // Otherwise, we use the normal method where the transform is called
  // for computing the transformation of every point.
  this->NonlinearThreadedGenerateData( outputRegionForThread, threadId );

} // end ThreadedGenerateData()


template< class TOutputImage, class TTransformPrecisionType >
void
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::NonlinearThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread,
  ThreadIdType threadId )
{
  // Get the output pointer
  OutputImagePointer outputPtr = this->GetOutput();

  // Create an iterator that will walk the output region for this thread.
  typedef ImageRegionIteratorWithIndex< TOutputImage > OutputIteratorType;
  OutputIteratorType it( outputPtr, outputRegionForThread );
  it.GoToBegin();

  // pixel coordinates
  PointType point;

  // Support for progress methods/callbacks
  ProgressReporter progress( this, threadId, outputRegionForThread.GetNumberOfPixels() );

  SpatialJacobianType sj;
  PixelType           sjOut;
  const unsigned int  nrElements = sj.GetVnlMatrix().size();

  // Walk the output region
  while( !it.IsAtEnd() )
  {
    // Determine the coordinates of the current voxel
    outputPtr->TransformIndexToPhysicalPoint( it.GetIndex(), point );

    this->m_Transform->GetSpatialJacobian( point, sj );

    // cast spatial jacobian to output pixel type
    vnl_copy( sj.GetVnlMatrix().begin(), sjOut.GetVnlMatrix().begin(),
      nrElements );

    // Set it
    it.Set( sjOut );

    // Update progress and iterator
    progress.CompletedPixel();
    ++it;
  }

} // end NonlinearThreadedGenerateData()


template< class TOutputImage, class TTransformPrecisionType >
void
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::LinearGenerateData( void )
{
  // Use an unthreaded implementation here, since the FillBuffer method
  // is used.

  // Get the output pointer
  OutputImagePointer outputPtr = this->GetOutput();

  // For linear transformation the spatial derivative is a constant,
  // i.e. it is independent of the spatial position.
  IndexType index; index.Fill( 1 );
  PointType point;
  outputPtr->TransformIndexToPhysicalPoint( index, point );

  SpatialJacobianType sj;
  PixelType           sjOut;
  const unsigned int  nrElements = sj.GetVnlMatrix().size();
  this->m_Transform->GetSpatialJacobian( point, sj );

  // cast spatial jacobian to output pixel type
  vnl_copy( sj.GetVnlMatrix().begin(), sjOut.GetVnlMatrix().begin(),
    nrElements );

  outputPtr->FillBuffer( sjOut );

} // end LinearThreadedGenerateData()


/**
 * Inform pipeline of required output region
 */
template< class TOutputImage, class TTransformPrecisionType >
void
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::GenerateOutputInformation( void )
{
  // call the superclass' implementation of this method
  Superclass::GenerateOutputInformation();

  // get pointer to the output
  OutputImagePointer outputPtr = this->GetOutput();
  if( !outputPtr )
  {
    return;
  }

  outputPtr->SetLargestPossibleRegion( m_OutputRegion );
  outputPtr->SetSpacing( m_OutputSpacing );
  outputPtr->SetOrigin( m_OutputOrigin );
  outputPtr->SetDirection( m_OutputDirection );
  outputPtr->Allocate();

} // end GenerateOutputInformation()


/**
 * Verify if any of the components has been modified.
 */
template< class TOutputImage, class TTransformPrecisionType >
unsigned long
TransformToSpatialJacobianSource< TOutputImage, TTransformPrecisionType >
::GetMTime( void ) const
{
  unsigned long latestTime = Object::GetMTime();

  if( this->m_Transform )
  {
    if( latestTime < this->m_Transform->GetMTime() )
    {
      latestTime = this->m_Transform->GetMTime();
    }
  }

  return latestTime;
} // end GetMTime()


} // end namespace itk

#endif // end #ifndef _itkTransformToSpatialJacobianSource_hxx
