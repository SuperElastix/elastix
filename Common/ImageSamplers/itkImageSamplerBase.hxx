/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __ImageSamplerBase_hxx
#define __ImageSamplerBase_hxx

#include "itkImageSamplerBase.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TInputImage >
ImageSamplerBase< TInputImage >
::ImageSamplerBase()
{
  this->m_Mask                      = 0;
  this->m_NumberOfMasks             = 0;
  this->m_NumberOfInputImageRegions = 0;
  this->m_NumberOfSamples           = 0;

  //tmp?
  this->m_UseMultiThread = false;

} // end Constructor()


/**
 * ******************* SetMask *******************
 */

template< class TInputImage >
void
ImageSamplerBase< TInputImage >
::SetMask( const MaskType * _arg, unsigned int pos )
{
  if( this->m_MaskVector.size() < pos + 1 )
  {
    this->m_MaskVector.resize( pos + 1 );
    this->m_NumberOfMasks = pos + 1;
  }
  if( pos == 0 )
  {
    this->m_Mask = _arg;
  }
  if( this->m_MaskVector[ pos ] != _arg )
  {
    this->m_MaskVector[ pos ] = _arg;

    /** The following line is not necessary, since the local
     * bounding box is already computed when SetImage() is called
     * in the elxRegistrationBase (when the mask spatial object
     * is constructed).
     */
    //this->m_Mask->ComputeLocalBoundingBox();
    this->Modified();
  }

} // SetMask()


/**
 * ******************* GetMask *******************
 */

template< class TInputImage >
const typename ImageSamplerBase< TInputImage >::MaskType
* ImageSamplerBase< TInputImage >
::GetMask( unsigned int pos ) const
{
  if( this->m_MaskVector.size() < pos + 1 )
  {
    return 0;
  }
  return this->m_MaskVector[ pos ];

} // end GetMask()

/**
 * ******************* SetNumberOfMasks *******************
 */

template< class TInputImage >
void
ImageSamplerBase< TInputImage >
::SetNumberOfMasks( const unsigned int _arg )
{
  if( this->m_NumberOfMasks != _arg )
  {
    this->m_MaskVector.resize( _arg );
    this->m_NumberOfMasks = _arg;
    this->Modified();
  }

} // end SetNumberOfMasks()


/**
 * ******************* SetInputImageRegion *******************
 */

template< class TInputImage >
void
ImageSamplerBase< TInputImage >
::SetInputImageRegion( const InputImageRegionType _arg, unsigned int pos )
{
  if( this->m_InputImageRegionVector.size() < pos + 1 )
  {
    this->m_InputImageRegionVector.resize( pos + 1 );
    this->m_NumberOfInputImageRegions = pos + 1;
  }
  if( pos == 0 )
  {
    this->m_InputImageRegion = _arg;
  }
  if( this->m_InputImageRegionVector[ pos ] != _arg )
  {
    this->m_InputImageRegionVector[ pos ] = _arg;
    this->Modified();
  }

} // SetInputImageRegion()


/**
 * ******************* GetInputImageRegion *******************
 */

template< class TInputImage >
const typename ImageSamplerBase< TInputImage >::InputImageRegionType
& ImageSamplerBase< TInputImage >
::GetInputImageRegion( unsigned int pos ) const
{
  if( this->m_InputImageRegionVector.size() < pos + 1 )
  {
    return this->m_DummyInputImageRegion;
  }
  return this->m_InputImageRegionVector[ pos ];

} // end GetInputImageRegion()

/**
 * ******************* SetNumberOfInputImageRegions *******************
 */

template< class TInputImage >
void
ImageSamplerBase< TInputImage >
::SetNumberOfInputImageRegions( const unsigned int _arg )
{
  if( this->m_NumberOfInputImageRegions != _arg )
  {
    this->m_InputImageRegionVector.resize( _arg );
    this->m_NumberOfInputImageRegions = _arg;
    this->Modified();
  }

} // end SetNumberOfInputImageRegions()


/**
 * ******************* GenerateInputRequestedRegion *******************
 */

template< class TInputImage >
void
ImageSamplerBase< TInputImage >
::GenerateInputRequestedRegion( void )
{
  /** Check if input image was set. */
  if( this->GetNumberOfInputs() == 0 )
  {
    itkExceptionMacro( << "ERROR: Input image not set" );
    return;
  }

  /** Get a pointer to the input image. */
  InputImagePointer inputImage = const_cast< InputImageType * >( this->GetInput() );

  /** Get and set the region. */
  if( this->GetInputImageRegion().GetNumberOfPixels() != 0 )
  {
    InputImageRegionType inputRequestedRegion = this->GetInputImageRegion();

    /** Crop the input requested region at the input's largest possible region. */
    if( inputRequestedRegion.Crop( inputImage->GetLargestPossibleRegion() ) )
    {
      inputImage->SetRequestedRegion( inputRequestedRegion );
    }
    else
    {
      /** Couldn't crop the region (requested region is outside the largest
      * possible region). Throw an exception.
      */

      /** Store what we tried to request (prior to trying to crop). */
      inputImage->SetRequestedRegion( inputRequestedRegion );

      /** Build an exception. */
      InvalidRequestedRegionError e( __FILE__, __LINE__ );
      e.SetLocation( ITK_LOCATION );
      e.SetDescription( "Requested region is (at least partially) outside the largest possible region." );
      e.SetDataObject( inputImage );
      throw e;
    }
  }
  else
  {
    inputImage->SetRequestedRegion( inputImage->GetLargestPossibleRegion() );
    this->SetInputImageRegion( inputImage->GetLargestPossibleRegion() );
  }

  /** Crop the region of the inputImage to the bounding box of the mask. */
  this->CropInputImageRegion();
  inputImage->SetRequestedRegion( this->m_CroppedInputImageRegion );

} // end GenerateInputRequestedRegion()


/**
 * ******************* SelectNewSamplesOnUpdate *******************
 */

template< class TInputImage >
bool
ImageSamplerBase< TInputImage >
::SelectNewSamplesOnUpdate( void )
{
  /** Set the Modified flag, such that on calling Update(),
  * the GenerateData method is executed again.
  * Return true to indicate that indeed new samples will be selected.
  * Inheriting subclasses may just return false and do nothing.
  */
  this->Modified();
  return true;

} // end SelectNewSamplesOnUpdate()


/**
 * ******************* IsInsideAllMasks *******************
 */

template< class TInputImage >
bool
ImageSamplerBase< TInputImage >
::IsInsideAllMasks( const InputImagePointType & point ) const
{
  bool ret = true;
  for( unsigned int i = 0; i < this->m_NumberOfMasks; ++i )
  {
    ret &= this->GetMask( i )->IsInside( point );
  }

  return ret;

} // end IsInsideAllMasks()


/**
 * ******************* UpdateAllMasks *******************
 */

template< class TInputImage >
void
ImageSamplerBase< TInputImage >
::UpdateAllMasks( void )
{
  /** If the masks are generated by a filter, then make sure they are updated. */
  for( unsigned int i = 0; i < this->m_NumberOfMasks; ++i )
  {
    if( this->GetMask( i )->GetSource() )
    {
      this->GetMask( i )->GetSource()->Update();
    }
  }

} // end UpdateAllMasks()


/**
 * ******************* CheckInputImageRegions *******************
 */

template< class TInputImage >
bool
ImageSamplerBase< TInputImage >
::CheckInputImageRegions( void )
{
  bool ret = true;
  for( unsigned int i = 0; i < this->GetNumberOfInputImageRegions(); ++i )
  {
    ret &= this->GetInput( i )->GetLargestPossibleRegion().IsInside(
      this->GetInputImageRegion( i ) );
  }
  return ret;

} // end CheckInputImageRegions()


/**
 * ******************* CropInputImageRegion *******************
 */

template< class TInputImage >
void
ImageSamplerBase< TInputImage >
::CropInputImageRegion( void )
{
  /** Since we expect to be called from GenerateInputRequestedRegion(),
   * we can safely assume that m_InputImageRegion is either
   * the LargestPossibleRegion of InputImage or a valid subregion of it.
   *
   * If a mask was set, then compute the intersection of the
   * InputImageRegion and the BoundingBoxRegion.
   */
  this->m_CroppedInputImageRegion = this->m_InputImageRegion;
  if( !this->m_Mask.IsNull() )
  {
    /** Get a handle to the input image. */
    InputImageConstPointer inputImage = this->GetInput();
    if( !inputImage )
    {
      return;
    }

    this->UpdateAllMasks();

    /** Get the indices of the bounding box extremes, based on the first mask.
     * Note that the bounding box is defined in terms of the mask
     * spacing and origin, and that we need a region in terms
     * of the inputImage indices.
     */

    typedef typename MaskType::BoundingBoxType        BoundingBoxType;
    typedef typename BoundingBoxType::PointsContainer PointsContainerType;
#if ITK_VERSION_MAJOR < 5
    typename BoundingBoxType::Pointer bb      = this->m_Mask->GetBoundingBox();
#else
    typename BoundingBoxType::ConstPointer bb = this->m_Mask->GetMyBoundingBoxInWorldSpace();
#endif
    typename BoundingBoxType::Pointer bbIndex = BoundingBoxType::New();
    const PointsContainerType * cornersWorld = bb->GetPoints();
    typename PointsContainerType::Pointer cornersIndex = PointsContainerType::New();
    cornersIndex->Reserve( cornersWorld->Size() );
    typename PointsContainerType::const_iterator itCW = cornersWorld->begin();
    typename PointsContainerType::iterator itCI       = cornersIndex->begin();
    typedef itk::ContinuousIndex<
      InputImagePointValueType, InputImageDimension > CIndexType;
    CIndexType cindex;
    while( itCW != cornersWorld->end() )
    {
      inputImage->TransformPhysicalPointToContinuousIndex( *itCW, cindex );
      *itCI = cindex;
      itCI++;
      itCW++;
    }
    bbIndex->SetPoints( cornersIndex );
    bbIndex->ComputeBoundingBox();

    /** Create a bounding box region. */
    InputImageIndexType minIndex, maxIndex;
    typedef typename InputImageIndexType::IndexValueType IndexValueType;
    InputImageSizeType   size;
    InputImageRegionType boundingBoxRegion;
    for( unsigned int i = 0; i < InputImageDimension; ++i )
    {
      /** apply ceil/floor for max/min resp. to be sure that
      * the bounding box is not too small */
      maxIndex[ i ] = static_cast< IndexValueType >(
        std::ceil( bbIndex->GetMaximum()[ i ] ) );
      minIndex[ i ] = static_cast< IndexValueType >(
        std::floor( bbIndex->GetMinimum()[ i ] ) );
      size[ i ] = maxIndex[ i ] - minIndex[ i ] + 1;
    }
    boundingBoxRegion.SetIndex( minIndex );
    boundingBoxRegion.SetSize( size );

    /** Compute the intersection. */
    bool cropped = this->m_CroppedInputImageRegion.Crop( boundingBoxRegion );

    /** If the cropping return false, then the intersection is empty.
     * In this case m_CroppedInputImageRegion is unchanged,
     * but we would like to throw an exception.
     */
    if( !cropped )
    {
      itkExceptionMacro( << "ERROR: the bounding box of the mask lies "
                         << "entirely out of the InputImageRegion!" );
    }
  }

} // end CropInputImageRegion()


/**
 * ******************* BeforeThreadedGenerateData *******************
 */

template< class TInputImage >
void
ImageSamplerBase< TInputImage >
::BeforeThreadedGenerateData( void )
{
  /** Initialize variables needed for threads. */
  this->m_ThreaderSampleContainer.clear();
  this->m_ThreaderSampleContainer.resize( this->GetNumberOfThreads() );
  for( std::size_t i = 0; i < this->GetNumberOfThreads(); i++ )
  {
    this->m_ThreaderSampleContainer[ i ] = ImageSampleContainerType::New();
  }

} // end BeforeThreadedGenerateData()


/**
 * ******************* AfterThreadedGenerateData *******************
 */

template< class TInputImage >
void
ImageSamplerBase< TInputImage >
::AfterThreadedGenerateData( void )
{
  /** Get the combined number of samples. */
  this->m_NumberOfSamples = 0;
  for( std::size_t i = 0; i < this->GetNumberOfThreads(); i++ )
  {
    this->m_NumberOfSamples += this->m_ThreaderSampleContainer[ i ]->Size();
  }

  /** Get handle to the output sample container. */
  typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();
  sampleContainer->clear();
  sampleContainer->reserve( this->m_NumberOfSamples );

  /** Combine the results of all threads. */
  for( std::size_t i = 0; i < this->GetNumberOfThreads(); i++ )
  {
    sampleContainer->insert( sampleContainer->end(),
      this->m_ThreaderSampleContainer[ i ]->begin(),
      this->m_ThreaderSampleContainer[ i ]->end() );
  }

} // end AfterThreadedGenerateData()


/**
 * ******************* PrintSelf *******************
 */

template< class TInputImage >
void
ImageSamplerBase< TInputImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "NumberOfMasks" << this->m_NumberOfMasks << std::endl;
  os << indent << "Mask: " << this->m_Mask.GetPointer() << std::endl;
  os << indent << "MaskVector:" << std::endl;
  for( unsigned int i = 0; i < this->m_NumberOfMasks; ++i )
  {
    os << indent.GetNextIndent() << this->m_MaskVector[ i ].GetPointer() << std::endl;
  }

  os << indent << "NumberOfInputImageRegions" << this->m_NumberOfInputImageRegions << std::endl;
  os << indent << "InputImageRegion: " << this->m_InputImageRegion << std::endl;
  os << indent << "InputImageRegionVector:" << std::endl;
  for( unsigned int i = 0; i < this->m_NumberOfInputImageRegions; ++i )
  {
    os << indent.GetNextIndent() << this->m_InputImageRegionVector[ i ] << std::endl;
  }
  os << indent << "CroppedInputImageRegion" << this->m_CroppedInputImageRegion << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef __ImageSamplerBase_hxx
