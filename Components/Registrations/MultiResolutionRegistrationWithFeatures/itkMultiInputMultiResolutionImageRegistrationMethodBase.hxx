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
#ifndef _itkMultiInputMultiResolutionImageRegistrationMethodBase_hxx
#define _itkMultiInputMultiResolutionImageRegistrationMethodBase_hxx

#include "itkMultiInputMultiResolutionImageRegistrationMethodBase.h"

#include "itkContinuousIndex.h"
#include "vnl/vnl_math.h"

/** macro that implements the Set methods */
#define itkImplementationSetMacro( _name, _type ) \
  template< typename TFixedImage, typename TMovingImage > \
  void \
  MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage > \
  ::Set##_name( _type _arg, unsigned int pos ) \
  { \
    if( pos == 0 ) \
    { \
      this->Superclass::Set##_name( _arg ); \
    } \
    if( pos >= this->GetNumberOf##_name##s() ) \
    { \
      this->SetNumberOf##_name##s( pos + 1 ); \
    } \
    if( this->m_##_name##s[ pos ] != _arg ) \
    { \
      this->m_##_name##s[ pos ] = _arg; \
      this->Modified(); \
    } \
  } // comment to allow ; after calling macro

/** macro that implements the Set methods */
#define itkImplementationSetMacro2( _name, _type ) \
  template< typename TFixedImage, typename TMovingImage > \
  void \
  MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage > \
  ::Set##_name( _type _arg, unsigned int pos ) \
  { \
    if( pos >= this->GetNumberOf##_name##s() ) \
    { \
      this->SetNumberOf##_name##s( pos + 1 ); \
    } \
    if( this->m_##_name##s[ pos ] != _arg ) \
    { \
      this->m_##_name##s[ pos ] = _arg; \
      this->Modified(); \
    } \
  } // comment to allow ; after calling macro

namespace itk
{
itkImplementationSetMacro( FixedImage, const FixedImageType * );
itkImplementationSetMacro( MovingImage, const MovingImageType * );
itkImplementationSetMacro( FixedImageRegion, FixedImageRegionType );
itkImplementationSetMacro( FixedImagePyramid, FixedImagePyramidType * );
itkImplementationSetMacro( MovingImagePyramid, MovingImagePyramidType * );
itkImplementationSetMacro( Interpolator, InterpolatorType * );
itkImplementationSetMacro2( FixedImageInterpolator, FixedImageInterpolatorType * );

/**
 * ****************** Constructor ******************
 */
template< typename TFixedImage, typename TMovingImage >
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::MultiInputMultiResolutionImageRegistrationMethodBase()
{}  // end Constructor()

/**
 * **************** GetFixedImage **********************************
 */

template< typename TFixedImage, typename TMovingImage >
const typename
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::FixedImageType
* MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::GetFixedImage( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfFixedImages() )
  {
    return 0;
  }
  else
  {
    return this->m_FixedImages[ pos ].GetPointer();
  }

}   // end GetFixedImage()

/**
 * **************** GetMovingImage **********************************
 */

template< typename TFixedImage, typename TMovingImage >
const typename
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::MovingImageType
* MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::GetMovingImage( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfMovingImages() )
  {
    return 0;
  }
  else
  {
    return this->m_MovingImages[ pos ].GetPointer();
  }

}   // end GetMovingImage()

/**
 * **************** GetInterpolator **********************************
 */

template< typename TFixedImage, typename TMovingImage >
typename
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::InterpolatorType
* MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::GetInterpolator( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfInterpolators() )
  {
    return 0;
  }
  else
  {
    return this->m_Interpolators[ pos ].GetPointer();
  }

}   // end GetInterpolator()

/**
 * **************** GetFixedImageInterpolator **********************************
 */

template< typename TFixedImage, typename TMovingImage >
typename
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::FixedImageInterpolatorType
* MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::GetFixedImageInterpolator( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfFixedImageInterpolators() )
  {
    return 0;
  }
  else
  {
    return this->m_FixedImageInterpolators[ pos ].GetPointer();
  }

}   // end GetFixedImageInterpolator()

/**
 * **************** GetFixedImagePyramid **********************************
 */

template< typename TFixedImage, typename TMovingImage >
typename
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::FixedImagePyramidType
* MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::GetFixedImagePyramid( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfFixedImagePyramids() )
  {
    return 0;
  }
  else
  {
    return this->m_FixedImagePyramids[ pos ].GetPointer();
  }

}   // end GetFixedImagePyramid()

/**
 * **************** GetMovingImagePyramid **********************************
 */

template< typename TFixedImage, typename TMovingImage >
typename
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::MovingImagePyramidType
* MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::GetMovingImagePyramid( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfMovingImagePyramids() )
  {
    return 0;
  }
  else
  {
    return this->m_MovingImagePyramids[ pos ].GetPointer();
  }

}   // end GetMovingImagePyramid()

/**
 * **************** GetFixedImageRegion **********************************
 */

template< typename TFixedImage, typename TMovingImage >
const typename
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::FixedImageRegionType
& MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::GetFixedImageRegion( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfFixedImageRegions() )
  {
    /** Return a dummy fixed image region */
    return this->m_NullFixedImageRegion;
  }
  else
  {
    return this->m_FixedImageRegions[ pos ];
  }

}   // end GetFixedImageRegion()

/*
 * ****************** SetMetric *******************************
 */

template< typename TFixedImage, typename TMovingImage >
void
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::SetMetric( MetricType * _arg )
{
  this->Superclass::SetMetric( _arg );

  MultiInputMetricType * testPointer = dynamic_cast< MultiInputMetricType * >( _arg );
  if( testPointer )
  {
    this->m_MultiInputMetric = testPointer;
  }
  else
  {
    itkExceptionMacro( << "ERROR: This registration method expects a MultiInputImageToImageMetric" );
  }

}   // end SetMetric()


/*
 * ****************** Initialize *******************************
 */

template< typename TFixedImage, typename TMovingImage >
void
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::Initialize() throw ( ExceptionObject )
{
  /** Sanity checks. */
  this->CheckOnInitialize();

  /** Setup the metric: the transform. */
  this->GetMultiInputMetric()->SetTransform( this->GetTransform() );

  /** Setup the metric: the images. */
  this->GetMultiInputMetric()->SetNumberOfFixedImages( this->GetNumberOfFixedImages() );
  this->GetMultiInputMetric()->SetNumberOfMovingImages( this->GetNumberOfMovingImages() );

  for( unsigned int i = 0; i < this->GetNumberOfFixedImagePyramids(); ++i )
  {
    this->GetMultiInputMetric()->SetFixedImage(
      this->GetFixedImagePyramid( i )->GetOutput( this->GetCurrentLevel() ), i );
  }

  for( unsigned int i = 0; i < this->GetNumberOfMovingImagePyramids(); ++i )
  {
    this->GetMultiInputMetric()->SetMovingImage(
      this->GetMovingImagePyramid( i )->GetOutput( this->GetCurrentLevel() ), i );
  }

  /** Setup the metric: the fixed image regions. */
  for( unsigned int i = 0; i < this->m_FixedImageRegionPyramids.size(); ++i )
  {
    this->GetMultiInputMetric()->SetFixedImageRegion(
      this->m_FixedImageRegionPyramids[ i ][ this->GetCurrentLevel() ], i );
  }

  /** Setup the metric: the interpolators. */
  for( unsigned int i = 0; i < this->GetNumberOfInterpolators(); ++i )
  {
    this->GetMultiInputMetric()->SetInterpolator( this->GetInterpolator( i ), i );
  }

  for( unsigned int i = 0; i < this->GetNumberOfFixedImageInterpolators(); ++i )
  {
    this->GetMultiInputMetric()
    ->SetFixedImageInterpolator( this->GetFixedImageInterpolator( i ), i );
  }

  /** Initialize the metric. */
  this->GetMultiInputMetric()->Initialize();

  /** Setup the optimizer. */
  this->GetOptimizer()->SetCostFunction( this->GetMetric() );
  this->GetOptimizer()->SetInitialPosition(
    this->GetInitialTransformParametersOfNextLevel() );

  /** Connect the transform to the Decorator. */
  TransformOutputType * transformOutput
    = static_cast< TransformOutputType * >( this->ProcessObject::GetOutput( 0 ) );

  transformOutput->Set( this->GetTransform() );

}   // end Initialize()


/*
 * ****************** PreparePyramids ******************
 */

template< typename TFixedImage, typename TMovingImage >
void
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::PreparePyramids( void )
{
  /** Check some assumptions. */
  this->CheckPyramids();

  /** Setup the moving image pyramids. */
  for( unsigned int i = 0; i < this->GetNumberOfMovingImagePyramids(); ++i )
  {
    MovingImagePyramidPointer movpyr = this->GetMovingImagePyramid( i );
    if( movpyr.IsNotNull() )
    {
      movpyr->SetNumberOfLevels( this->GetNumberOfLevels() );
      if( this->GetNumberOfMovingImages() > 1 )
      {
        movpyr->SetInput( this->GetMovingImage( i ) );
      }
      else
      {
        movpyr->SetInput( this->GetMovingImage() );
      }
      movpyr->UpdateLargestPossibleRegion();
    }
  }

  /** Setup the fixed image pyramids and the fixed image region pyramids. */
  typedef typename FixedImageRegionType::SizeType      SizeType;
  typedef typename FixedImageRegionType::IndexType     IndexType;
  typedef typename FixedImagePyramidType::ScheduleType ScheduleType;

  this->m_FixedImageRegionPyramids.resize( this->GetNumberOfFixedImagePyramids() );

  for( unsigned int i = 0; i < this->GetNumberOfFixedImagePyramids(); ++i )
  {
    /** Setup the fixed image pyramid. */
    FixedImagePyramidPointer fixpyr = this->GetFixedImagePyramid( i );
    if( fixpyr.IsNotNull() )
    {
      fixpyr->SetNumberOfLevels( this->GetNumberOfLevels() );
      if( this->GetNumberOfFixedImages() > 1 )
      {
        fixpyr->SetInput( this->GetFixedImage( i ) );
      }
      else
      {
        fixpyr->SetInput( this->GetFixedImage() );
      }
      fixpyr->UpdateLargestPossibleRegion();

      /** Setup the fixed image region pyramid. */
      ScheduleType schedule = fixpyr->GetSchedule();

      FixedImageRegionType fixedImageRegion;
      if( this->GetNumberOfFixedImageRegions() > 1 )
      {
        fixedImageRegion = this->GetFixedImageRegion( i );
      }
      else
      {
        fixedImageRegion = this->GetFixedImageRegion();
      }
      SizeType  inputSize  = fixedImageRegion.GetSize();
      IndexType inputStart = fixedImageRegion.GetIndex();
      IndexType inputEnd   = inputStart;
      for( unsigned int dim = 0; dim < TFixedImage::ImageDimension; dim++ )
      {
        inputEnd[ dim ] += ( inputSize[ dim ] - 1 );
      }

      this->m_FixedImageRegionPyramids[ i ].reserve( this->GetNumberOfLevels() );
      this->m_FixedImageRegionPyramids[ i ].resize( this->GetNumberOfLevels() );

      // Compute the FixedImageRegion corresponding to each level of the
      // pyramid.
      //
      // In the ITK implementation this uses the same algorithm of the ShrinkImageFilter
      // since the regions should be compatible. However, we inherited another
      // Multiresolution pyramid, which does not use the same shrinking pattern.
      // Instead of copying the shrinking code, we compute image regions from
      // the result of the fixed image pyramid.
      typedef typename FixedImageType::PointType                           PointType;
      typedef typename PointType::CoordRepType                             CoordRepType;
      typedef typename IndexType::IndexValueType                           IndexValueType;
      typedef typename SizeType::SizeValueType                             SizeValueType;
      typedef ContinuousIndex< CoordRepType, TFixedImage::ImageDimension > CIndexType;

      PointType inputStartPoint;
      PointType inputEndPoint;
      fixpyr->GetInput()->TransformIndexToPhysicalPoint( inputStart, inputStartPoint );
      fixpyr->GetInput()->TransformIndexToPhysicalPoint( inputEnd, inputEndPoint );

      for( unsigned int level = 0; level < this->GetNumberOfLevels(); level++ )
      {
        SizeType         size;
        IndexType        start;
        CIndexType       startcindex;
        CIndexType       endcindex;
        FixedImageType * fixedImageAtLevel = fixpyr->GetOutput( level );
        /** Map the original fixed image region to the image resulting from the
         * FixedImagePyramid at level l.
         * To be on the safe side, the start point is ceiled, and the end point is
         * floored. To see why, consider an image of 4 by 4, and its downsampled version of 2 by 2. */
        fixedImageAtLevel->TransformPhysicalPointToContinuousIndex( inputStartPoint, startcindex );
        fixedImageAtLevel->TransformPhysicalPointToContinuousIndex( inputEndPoint, endcindex );
        for( unsigned int dim = 0; dim < TFixedImage::ImageDimension; dim++ )
        {
          start[ dim ] = static_cast< IndexValueType >( vcl_ceil( startcindex[ dim ] ) );
          size[ dim ]  = static_cast< SizeValueType >(
            static_cast< SizeValueType >( vcl_floor( endcindex[ dim ] ) ) - start[ dim ] + 1 );
        }

        this->m_FixedImageRegionPyramids[ i ][ level ].SetSize( size );
        this->m_FixedImageRegionPyramids[ i ][ level ].SetIndex( start );

      }   // end for loop over res levels

    }   // end if fixpyr!=0

  }   // end for loop over fixed pyramids

}   // end PreparePyramids()


/*
 * ********************* GenerateData ***********************
 */

template< typename TFixedImage, typename TMovingImage >
void
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::GenerateData()
{
  this->m_Stop = false;

  /** Check the transform and set the initial parameters. */
  if( this->GetTransform() == 0 )
  {
    itkExceptionMacro( << "Transform is not present" );
  }

  this->SetInitialTransformParametersOfNextLevel( this->GetInitialTransformParameters() );

  if( this->GetInitialTransformParametersOfNextLevel().Size() !=
    this->GetTransform()->GetNumberOfParameters() )
  {
    itkExceptionMacro( << "Size mismatch between initial parameter and transform" );
  }

  /** Prepare the fixed and moving pyramids. */
  this->PreparePyramids();

  /** Loop over the resolution levels. */
  for( unsigned int currentLevel = 0; currentLevel < this->GetNumberOfLevels(); currentLevel++ )
  {
    this->SetCurrentLevel( currentLevel );

    // Invoke an iteration event.
    // This allows a UI to reset any of the components between
    // resolution level.
    this->InvokeEvent( IterationEvent() );

    // Check if there has been a stop request
    if( this->m_Stop )
    {
      break;
    }

    try
    {
      // initialize the interconnects between components
      this->Initialize();
    }
    catch( ExceptionObject & err )
    {
      this->m_LastTransformParameters = ParametersType( 1 );
      this->m_LastTransformParameters.Fill( 0.0f );

      // pass exception to caller
      throw err;
    }

    try
    {
      // do the optimization
      this->GetOptimizer()->StartOptimization();
    }
    catch( ExceptionObject & err )
    {
      // An error has occurred in the optimization.
      // Update the parameters
      this->m_LastTransformParameters = this->GetOptimizer()->GetCurrentPosition();

      // Pass exception to caller
      throw err;
    }

    /** Get the results. */
    this->m_LastTransformParameters = this->GetOptimizer()->GetCurrentPosition();
    this->GetTransform()->SetParameters( this->m_LastTransformParameters );

    /** Setup the initial parameters for next level. */
    if( this->GetCurrentLevel() < this->GetNumberOfLevels() - 1 )
    {
      this->SetInitialTransformParametersOfNextLevel(
        this->m_LastTransformParameters );
    }

  }   // end for loop over res levels

}   // end GenerateData()


/**
 * ***************** GetMTime ******************
 */

template< typename TFixedImage, typename TMovingImage >
unsigned long
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::GetMTime() const
{
  unsigned long mtime = Superclass::GetMTime();
  unsigned long m;

  // Some of the following should be removed once ivars are put in the
  // input and output lists

  for( unsigned int i = 0; i < this->GetNumberOfInterpolators(); ++i )
  {
    InterpolatorPointer interpolator = this->GetInterpolator( i );
    if( interpolator )
    {
      m     = interpolator->GetMTime();
      mtime = ( m > mtime ? m : mtime );
    }
  }

  for( unsigned int i = 0; i < this->GetNumberOfFixedImages(); ++i )
  {
    FixedImageConstPointer fixedImage = this->GetFixedImage( i );
    if( fixedImage )
    {
      m     = fixedImage->GetMTime();
      mtime = ( m > mtime ? m : mtime );
    }
  }

  for( unsigned int i = 0; i < this->GetNumberOfMovingImages(); ++i )
  {
    MovingImageConstPointer movingImage = this->GetMovingImage( i );
    if( movingImage )
    {
      m     = movingImage->GetMTime();
      mtime = ( m > mtime ? m : mtime );
    }
  }

  for( unsigned int i = 0; i < this->GetNumberOfFixedImagePyramids(); ++i )
  {
    FixedImagePyramidPointer fixedImagePyramid = this->GetFixedImagePyramid( i );
    if( fixedImagePyramid )
    {
      m     = fixedImagePyramid->GetMTime();
      mtime = ( m > mtime ? m : mtime );
    }
  }

  for( unsigned int i = 0; i < this->GetNumberOfMovingImagePyramids(); ++i )
  {
    MovingImagePyramidPointer movingImagePyramid = this->GetMovingImagePyramid( i );
    if( movingImagePyramid )
    {
      m     = movingImagePyramid->GetMTime();
      mtime = ( m > mtime ? m : mtime );
    }
  }

  return mtime;

}   // end GetMTime()


/*
 * ****************** CheckPyramids ******************
 */

template< typename TFixedImage, typename TMovingImage >
void
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::CheckPyramids( void ) throw ( ExceptionObject )
{
  /** Check if at least one of the following are provided. */
  if( this->GetFixedImage() == 0 )
  {
    itkExceptionMacro( << "FixedImage is not present" );
  }
  if( this->GetMovingImage() == 0 )
  {
    itkExceptionMacro( << "MovingImage is not present" );
  }
  if( this->GetFixedImagePyramid() == 0 )
  {
    itkExceptionMacro( << "Fixed image pyramid is not present" );
  }
  if( this->GetMovingImagePyramid() == 0 )
  {
    itkExceptionMacro( << "Moving image pyramid is not present" );
  }

  /** Check if the number if fixed/moving pyramids >= nr of fixed/moving images,
   * and whether the number of fixed image regions == the number of fixed images.
   */
  if( this->GetNumberOfFixedImagePyramids() < this->GetNumberOfFixedImages() )
  {
    itkExceptionMacro( << "The number of fixed image pyramids should be >= the number of fixed images" );
  }
  if( this->GetNumberOfMovingImagePyramids() < this->GetNumberOfMovingImages() )
  {
    itkExceptionMacro( << "The number of moving image pyramids should be >= the number of moving images" );
  }
  if( this->GetNumberOfFixedImageRegions() != this->GetNumberOfFixedImages() )
  {
    itkExceptionMacro( << "The number of fixed image regions should equal the number of fixed image" );
  }

}   // end CheckPyramids()


/*
 * ****************** CheckOnInitialize ******************
 */

template< typename TFixedImage, typename TMovingImage >
void
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::CheckOnInitialize( void ) throw ( ExceptionObject )
{
  /** check if at least one of the following is present. */
  if( this->GetMetric() == 0 )
  {
    itkExceptionMacro( << "Metric is not present" );
  }
  if( this->GetOptimizer() == 0 )
  {
    itkExceptionMacro( << "Optimizer is not present" );
  }
  if( this->GetTransform() == 0 )
  {
    itkExceptionMacro( << "Transform is not present" );
  }
  if( this->GetInterpolator() == 0 )
  {
    itkExceptionMacro( << "Interpolator is not present" );
  }

  /** nrofinterpolators >= nrofpyramids? */
  if( this->GetNumberOfMovingImagePyramids() >
    this->GetNumberOfInterpolators() )
  {
    itkExceptionMacro( << "NumberOfMovingImagePyramids can not exceed the NumberOfInterpolators!" );
  }

}   // end CheckOnInitialize()


/*
 * ****************** PrintSelf ******************
 */

template< typename TFixedImage, typename TMovingImage >
void
MultiInputMultiResolutionImageRegistrationMethodBase< TFixedImage, TMovingImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  /** Print all fixed images. */
  os << indent << "Fixed images: [ ";
  for( unsigned int i = 0; i < this->GetNumberOfFixedImages(); ++i )
  {
    os << this->m_FixedImages[ i ] << " ";
  }
  os << "]" << std::endl;

  /** Print all moving images. */
  os << indent << "Moving images: [ ";
  for( unsigned int i = 0; i < this->GetNumberOfMovingImages(); ++i )
  {
    os << this->m_MovingImages[ i ] << " ";
  }
  os << "]" << std::endl;

  /** Print all fixed image regions. */
  os << indent << "FixedImageRegions: [ ";
  for( unsigned int i = 0; i < this->GetNumberOfFixedImageRegions(); ++i )
  {
    os << this->m_FixedImageRegions[ i ] << " ";
  }
  os << "]" << std::endl;

  /** Print all fixed image region pyramids. */
  os << indent << "FixedImageRegionPyramids: [ ";
  for( unsigned int i = 0; i < this->GetNumberOfFixedImageRegions(); ++i )
  {
    os << " [ ";
    for( unsigned int j = 0; j < this->m_FixedImageRegionPyramids[ i ].size(); ++j )
    {
      os << this->m_FixedImageRegionPyramids[ i ][ j ] << " ";
    }
    os << "]";
  }
  os << " ]" << std::endl;

  /** Print all fixed image pyramids. */
  os << indent << "FixedImagePyramids: [ ";
  for( unsigned int i = 0; i < this->GetNumberOfFixedImagePyramids(); ++i )
  {
    os << this->m_FixedImagePyramids[ i ] << " ";
  }
  os << "]" << std::endl;

  /** Print all moving image pyramids. */
  os << indent << "MovingImagePyramids: [ ";
  for( unsigned int i = 0; i < this->GetNumberOfMovingImagePyramids(); ++i )
  {
    os << this->m_MovingImagePyramids[ i ] << " ";
  }
  os << "]" << std::endl;

  /** Print all moving image interpolators. */
  os << indent << "Interpolators: [ ";
  for( unsigned int i = 0; i < this->GetNumberOfInterpolators(); ++i )
  {
    os << this->m_Interpolators[ i ] << " ";
  }
  os << "]" << std::endl;

  /** Print all fixed image interpolators. */
  os << indent << "FixedImageInterpolators: [ ";
  for( unsigned int i = 0; i < this->GetNumberOfFixedImageInterpolators(); ++i )
  {
    os << this->m_FixedImageInterpolators[ i ] << " ";
  }
  os << "]" << std::endl;

}   // end PrintSelf()


} // end namespace itk

#undef itkImplementationSetMacro
#undef itkImplementationSetMacro2

#endif // end #ifndef _itkMultiInputMultiResolutionImageRegistrationMethodBase_hxx
