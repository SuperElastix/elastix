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
#ifndef __itkGenericMultiResolutionPyramidImageFilter_hxx
#define __itkGenericMultiResolutionPyramidImageFilter_hxx

#include "itkGenericMultiResolutionPyramidImageFilter.h"

#include "itkResampleImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkImageAlgorithm.h"

namespace // anonymous namespace
{
/**
 * ******************* UpdateAndGraft ***********************
 */

template< class GenericMultiResolutionPyramidImageFilterType,
class ImageToImageFilterType, typename OutputImageType >
void
UpdateAndGraft(
  typename GenericMultiResolutionPyramidImageFilterType::Pointer thisFilter,
  typename ImageToImageFilterType::Pointer & filter,
  OutputImageType * outImage, const unsigned int ilevel )
{
  filter->GraftOutput( outImage );

  // force to always update in case shrink factors are the same
  filter->Modified();
  filter->UpdateLargestPossibleRegion();
  thisFilter->GraftNthOutput( ilevel, filter->GetOutput() );
} // end UpdateAndGraft()


} // end namespace anonymous

namespace itk
{
/**
 * ******************* Constructor ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::GenericMultiResolutionPyramidImageFilter()
{
  this->m_CurrentLevel               = 0;
  this->m_ComputeOnlyForCurrentLevel = false;
  SmoothingScheduleType temp( this->GetNumberOfLevels(), ImageDimension );
  temp.Fill( NumericTraits< ScalarRealType >::ZeroValue() );
  this->m_SmoothingSchedule        = temp;
  this->m_SmoothingScheduleDefined = false;
} // end Constructor


/**
 * ******************* SetNumberOfLevels ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::SetNumberOfLevels( unsigned int num )
{
  if( this->m_NumberOfLevels == num ) { return; }
  Superclass::SetNumberOfLevels( num );

  /** Resize the smoothing schedule too. */
  SmoothingScheduleType temp( this->m_NumberOfLevels, ImageDimension );
  temp.Fill( NumericTraits< ScalarRealType >::ZeroValue() );
  this->m_SmoothingSchedule        = temp;
  this->m_SmoothingScheduleDefined = false;

} // end SetNumberOfLevels()


/**
 * ******************* SetCurrentLevel ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::SetCurrentLevel( unsigned int level )
{
  itkDebugMacro( "setting CurrentLevel to " << level );
  if( this->m_CurrentLevel != level )
  {
    // clamp value to be less then number of levels
    this->m_CurrentLevel = level;
    if( this->m_CurrentLevel >= this->m_NumberOfLevels )
    {
      // Safe this->m_NumberOfLevels always >= 1
      this->m_CurrentLevel = this->m_NumberOfLevels - 1;
    }
    this->ReleaseOutputs();

    /** Only set the modified flag for this filter if the output is computed per level. */
    if( this->m_ComputeOnlyForCurrentLevel )
    {
      this->Modified();
    }
  }
} // end SetCurrentLevel()


/**
 * ******************* SetComputeOnlyForCurrentLevel ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::SetComputeOnlyForCurrentLevel( const bool _arg )
{
  itkDebugMacro( "setting ComputeOnlyForCurrentLevel to " << _arg );
  if( this->m_ComputeOnlyForCurrentLevel != _arg )
  {
    this->m_ComputeOnlyForCurrentLevel = _arg;
    this->ReleaseOutputs();
    this->Modified();
  }
} // end SetComputeOnlyForCurrentLevel()


/**
 * ******************* SetSchedule ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::SetSchedule( const ScheduleType & schedule )
{
  Superclass::SetSchedule( schedule );

  /** This part is to make sure that only combination of
   * SetRescaleSchedule and SetSmoothingSchedule or SetSchedule are used.
   * Only in GenerateData we use SetSmoothingScheduleToDefault, because only
   * there all required information is available.
   */
  SmoothingScheduleType temp( this->GetNumberOfLevels(), ImageDimension );
  temp.Fill( 0 );
  this->m_SmoothingSchedule        = temp;
  this->m_SmoothingScheduleDefined = false;
} // end SetSchedule()


/**
 * ******************* SetRescaleSchedule ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::SetRescaleSchedule( const RescaleScheduleType & schedule )
{
  /** Here we would prefer to use m_RescaleSchedule.
   * Although it would require copying most of the methods
   * from MultiResolutionPyramidImageFilter and changing m_Schedule
   * to m_RescaleSchedule.
   */
  Superclass::SetSchedule( schedule );
} // end SetRescaleSchedule()


/**
 * ******************* SetRescaleScheduleToUnity ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::SetRescaleScheduleToUnity( void )
{
  RescaleScheduleType schedule;
  schedule.Fill( NumericTraits< ScalarRealType >::OneValue() );
  Superclass::SetSchedule( schedule );
} // end SetRescaleScheduleToUnity()


/**
 * ******************* SetSmoothingSchedule ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::SetSmoothingSchedule( const SmoothingScheduleType & schedule )
{
  if( schedule == this->m_SmoothingSchedule )
  {
    return;
  }

  if( schedule.rows() != this->m_NumberOfLevels
    || schedule.columns() != ImageDimension )
  {
    itkDebugMacro( << "Smoothing schedule has wrong dimensions" );
    return;
  }

  for( unsigned int level = 0; level < this->m_NumberOfLevels; level++ )
  {
    for( unsigned int dim = 0; dim < ImageDimension; dim++ )
    {
      this->m_SmoothingSchedule[ level ][ dim ] = schedule[ level ][ dim ];

      /** Similar to Superclass::SetSchedule, set smoothing schedule to
       * max( 0, min(schedule[level], schedule[level-1] ).
       */
      if( level > 0 )
      {
        this->m_SmoothingSchedule[ level ][ dim ] = vnl_math_min(
          this->m_SmoothingSchedule[ level ][ dim ],
          this->m_SmoothingSchedule[ level - 1 ][ dim ] );
      }
      if( this->m_SmoothingSchedule[ level ][ dim ] < 0.0 )
      {
        this->m_SmoothingSchedule[ level ][ dim ] = 0.0;
      }
    }
  }

  this->m_SmoothingScheduleDefined = true;
  this->Modified();
} // end SetSmoothingSchedule()


/**
 * ******************* SetSmoothingScheduleToZero ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::SetSmoothingScheduleToZero( void )
{
  SmoothingScheduleType schedule;
  schedule.Fill( NumericTraits< ScalarRealType >::ZeroValue() );
  this->SetSmoothingSchedule( schedule );
} // end SetSmoothingScheduleToZero()


/**
 * ******************* GenerateData ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::GenerateData( void )
{
  // Depending on user setting of the SetUseMultiResolutionRescaleSchedule() and
  // SetUseMultiResolutionSmoothingSchedule()
  // in combination with SetUseShrinkImageFilter() different pipelines will be
  // created below. The most common is:
  // 1. m_UseMultiResolutionSmoothingSchedule = true
  //    m_UseMultiResolutionRescaleSchedule = true
  //    Then pipeline is: input -> smoother -> shrinker/resample -> output
  // 2. m_UseMultiResolutionSmoothingSchedule = false
  //    m_UseMultiResolutionRescaleSchedule = true
  //    Then pipeline is: input -> shrinker/resample -> output
  // 3. m_UseMultiResolutionSmoothingSchedule = true
  //    m_UseMultiResolutionRescaleSchedule = false
  //    Then pipeline is: input -> smoother -> output
  // 4. m_UseMultiResolutionSmoothingSchedule = false
  //    m_UseMultiResolutionRescaleSchedule = false
  //    Then pipeline is: input -> copy -> output
  //
  // 1.a) The smoother can be skipped if AreSigmasAllZeros(...)
  //      returns true for the current level.
  // 1.b) The shrinker/resampler can be skipped if AreRescaleFactorsAllOnes(...)
  //      returns true for the current level.
  //
  // Then pipeline 1 may transforms for the current level to:
  // 1.a) input -> shrinker/resample -> output
  // 1.b) input -> smoother -> output
  //
  // Pipeline also takes care of memory allocation for N'th output if
  // SetComputeOnlyForCurrentLevel has been set to true.

  // Get the input and output pointers
  InputImageConstPointer input = this->GetInput();

  // Check if we have to do anything at all
  if( !this->IsSmoothingUsed() && !this->IsRescaleUsed() )
  {
    // This is a special case we just allocate output images and copy input
    for( unsigned int level = 0; level < this->m_NumberOfLevels; ++level )
    {
      if( !this->m_ComputeOnlyForCurrentLevel )
      {
        this->UpdateProgress( static_cast< float >( level )
          / static_cast< float >( this->m_NumberOfLevels ) );
      }

      if( this->ComputeForCurrentLevel( level ) )
      {
        OutputImagePointer outputPtr = this->GetOutput( level );
        outputPtr->SetBufferedRegion( input->GetLargestPossibleRegion() );
        outputPtr->Allocate();

        ImageAlgorithm::Copy( input.GetPointer(), outputPtr.GetPointer(),
          input->GetLargestPossibleRegion(), outputPtr->GetLargestPossibleRegion() );
      }
    }
    return; // We are done, return
  }

  // First check if smoothing schedule has been set
  if( !this->m_SmoothingScheduleDefined )
  {
    this->SetSmoothingScheduleToDefault();
  }

  typename SmootherType::Pointer smoother;
  typename ImageToImageFilterSameTypes::Pointer rescaleSameTypes;
  typename ImageToImageFilterDifferentTypes::Pointer rescaleDifferentTypes;

  for( unsigned int level = 0; level < this->m_NumberOfLevels; ++level )
  {
    if( !this->m_ComputeOnlyForCurrentLevel )
    {
      this->UpdateProgress( static_cast< float >( level )
        / static_cast< float >( this->m_NumberOfLevels ) );
    }

    if( this->ComputeForCurrentLevel( level ) )
    {
      // Allocate memory for each output
      OutputImagePointer outputPtr = this->GetOutput( level );
      outputPtr->SetBufferedRegion( outputPtr->GetRequestedRegion() );
      outputPtr->Allocate();

      // Setup the smoother
      const bool smootherIsUsed = this->SetupSmoother( level, smoother, input );

      // Setup the shrinker or resampler
      const int shrinkerOrResamplerIsUsed = this->SetupShrinkerOrResampler( level,
        smoother, smootherIsUsed, input, outputPtr,
        rescaleSameTypes, rescaleDifferentTypes );

      // Update the pipeline and graft or copy results to this filters output
      if( shrinkerOrResamplerIsUsed == 0 && smootherIsUsed )
      {
        UpdateAndGraft< Self, SmootherType, OutputImageType >(
          this, smoother, outputPtr, level );
      }
      else if( shrinkerOrResamplerIsUsed == 0 )
      {
        ImageAlgorithm::Copy( input.GetPointer(), outputPtr.GetPointer(),
          input->GetLargestPossibleRegion(), outputPtr->GetLargestPossibleRegion() );
      }
      else if( shrinkerOrResamplerIsUsed == 1 )
      {
        UpdateAndGraft< Self, ImageToImageFilterSameTypes, OutputImageType >(
          this, rescaleSameTypes, outputPtr, level );
      }
      else if( shrinkerOrResamplerIsUsed == 2 )
      {
        UpdateAndGraft< Self, ImageToImageFilterDifferentTypes, OutputImageType >(
          this, rescaleDifferentTypes, outputPtr, level );
      }
      // no else needed

    }
  } // end for ilevel
}   // end GenerateData()


/**
 * ******************* SetupSmoother ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
bool
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::SetupSmoother( const unsigned int level,
  typename SmootherType::Pointer & smoother,
  const InputImageConstPointer & input )
{
  SigmaArrayType sigmaArray;
  this->GetSigma( level, sigmaArray );
  const bool sigmasAllZeros = this->AreSigmasAllZeros( sigmaArray );
  if( !sigmasAllZeros )
  {
    // First construct the smoother if has not been created and set input.
    if( smoother.IsNull() ) { smoother = SmootherType::New(); }

    smoother->SetInput( input );
    smoother->SetSigmaArray( sigmaArray );
    return true;
  }

  return false;
} // end SetupSmoother()


/**
 * ******************* SetupShrinkerOrResampler ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
int
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::SetupShrinkerOrResampler( const unsigned int level,
  typename SmootherType::Pointer & smoother, const bool sameType,
  const InputImageConstPointer & inputPtr,
  const OutputImagePointer & outputPtr,
  typename ImageToImageFilterSameTypes::Pointer & rescaleSameTypes,
  typename ImageToImageFilterDifferentTypes::Pointer & rescaleDifferentTypes )
{
  RescaleFactorArrayType shrinkFactors;
  this->GetShrinkFactors( level, shrinkFactors );
  const bool rescaleFactorsAllOnes = this->AreRescaleFactorsAllOnes( shrinkFactors );

  // No shrinking or resampling needed: return 0
  if( rescaleFactorsAllOnes ) { return 0; }

  // Choose between shrinker or resampler
  this->DefineShrinkerOrResampler( sameType, shrinkFactors, outputPtr,
    rescaleSameTypes, rescaleDifferentTypes );

  // Rescaling is done with input and output type being equal: return 1
  // Input and output are equal only if the smoother was used previously.
  if( sameType )
  {
    rescaleSameTypes->SetInput( smoother->GetOutput() );
    return 1;
  }

  // Rescaling is done with input and output type being different: return 2
  // Input and output are different only if the smoother was skipped.
  rescaleDifferentTypes->SetInput( inputPtr );
  return 2;

} // end SetupShrinkerOrResampler()


/**
 * ******************* DefineShrinkerOrResampler ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::DefineShrinkerOrResampler( const bool sameType,
  const RescaleFactorArrayType & shrinkFactors,
  const OutputImagePointer & outputPtr,
  typename ImageToImageFilterSameTypes::Pointer & rescaleSameTypes,
  typename ImageToImageFilterDifferentTypes::Pointer & rescaleDifferentTypes )
{
  // Typedefs
  typedef IdentityTransform< TPrecisionType, OutputImageType::ImageDimension >    TransformType;
  typedef ShrinkImageFilter< OutputImageType, OutputImageType >                   ShrinkerSameType;
  typedef ResampleImageFilter< OutputImageType, OutputImageType, TPrecisionType > ResamplerSameType;
  typedef ShrinkImageFilter< InputImageType, OutputImageType >                    ShrinkerDifferentType;
  typedef ResampleImageFilter< InputImageType, OutputImageType, TPrecisionType >  ResamplerDifferentType;
  typedef LinearInterpolateImageFunction< OutputImageType, TPrecisionType >       InterpolatorForSameType;
  typedef LinearInterpolateImageFunction< InputImageType, TPrecisionType >        InterpolatorForDifferentType;

  /**
   * Define pipeline in case input and output types are THE SAME.
   */

  if( sameType )
  {
    // A pipeline version that newly constructs the required filters:
    if( rescaleSameTypes.IsNull() )
    {
      if( this->GetUseShrinkImageFilter() )
      {
        // Define and setup shrinker
        typename ShrinkerSameType::Pointer shrinker = ShrinkerSameType::New();
        shrinker->SetShrinkFactors( shrinkFactors );

        // Assign
        rescaleSameTypes = shrinker.GetPointer();
      }
      else
      {
        // Define and setup resampler
        typename ResamplerSameType::Pointer resampler = ResamplerSameType::New();
        resampler->SetOutputParametersFromImage( outputPtr );
        resampler->SetDefaultPixelValue( 0 );

        // Define and set interpolator
        typename InterpolatorForSameType::Pointer interpolator = InterpolatorForSameType::New();
        resampler->SetInterpolator( interpolator );

        // Define and set transform
        typename TransformType::Pointer transform = TransformType::New();
        resampler->SetTransform( transform );

        // Assign
        rescaleSameTypes = resampler.GetPointer();
      }
    }
    // A pipeline version that re-uses previously constructed filters:
    else
    {
      if( this->GetUseShrinkImageFilter() )
      {
        // Setup shrinker
        typename ShrinkerSameType::Pointer shrinker
          = dynamic_cast< ShrinkerSameType * >( rescaleSameTypes.GetPointer() );
        shrinker->SetShrinkFactors( shrinkFactors );
      }
      else
      {
        // Setup resampler
        typename ResamplerSameType::Pointer resampler
          = dynamic_cast< ResamplerSameType * >( rescaleSameTypes.GetPointer() );
        resampler->SetOutputParametersFromImage( outputPtr );
      }
    }

    return;
  }

  /**
   * Define pipeline in case input and output types are DIFFERENT.
   */

  // A pipeline version that newly constructs the required filters:
  if( rescaleDifferentTypes.IsNull() )
  {
    if( this->GetUseShrinkImageFilter() )
    {
      // Define and setup shrinker
      typename ShrinkerDifferentType::Pointer shrinker = ShrinkerDifferentType::New();
      shrinker->SetShrinkFactors( shrinkFactors );

      // Assign
      rescaleDifferentTypes = shrinker.GetPointer();
    }
    else
    {
      // Define and setup resampler
      typename ResamplerDifferentType::Pointer resampler = ResamplerDifferentType::New();
      resampler->SetOutputParametersFromImage( outputPtr );
      resampler->SetDefaultPixelValue( 0 );

      // Define and set interpolator
      typename InterpolatorForDifferentType::Pointer interpolator = InterpolatorForDifferentType::New();
      resampler->SetInterpolator( interpolator );

      // Define and set transform
      typename TransformType::Pointer transform = TransformType::New();
      resampler->SetTransform( transform );

      // Assign
      rescaleDifferentTypes = resampler.GetPointer();
    }
  }
  // A pipeline version that re-uses previously constructed filters:
  else
  {
    if( this->GetUseShrinkImageFilter() )
    {
      typename ShrinkerDifferentType::Pointer shrinker
        = dynamic_cast< ShrinkerDifferentType * >( rescaleDifferentTypes.GetPointer() );
      shrinker->SetShrinkFactors( shrinkFactors );
    }
    else
    {
      typename ResamplerDifferentType::Pointer resampler
        = dynamic_cast< ResamplerDifferentType * >( rescaleDifferentTypes.GetPointer() );
      resampler->SetOutputParametersFromImage( outputPtr );
    }
  }

} // end DefineShrinkerOrResampler()


/**
 * ******************* GenerateOutputInformation ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::GenerateOutputInformation( void )
{
  if( this->IsRescaleUsed() )
  {
    Superclass::GenerateOutputInformation();
  }
  else
  {
    // call the SuperSuperclass implementation of this method
    SuperSuperclass::GenerateOutputInformation();
  }
} // end GenerateOutputInformation()


/**
 * ******************* GenerateOutputRequestedRegion ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::GenerateOutputRequestedRegion( DataObject * refOutput )
{
  if( this->IsRescaleUsed() )
  {
    Superclass::GenerateOutputRequestedRegion( refOutput );
  }
  else
  {
    // call the supersuperclass's implementation of this method
    SuperSuperclass::GenerateOutputRequestedRegion( refOutput );
  }

  // We have to set requestedRegion properly
  for( unsigned int level = 0; level < this->m_NumberOfLevels; level++ )
  {
    this->GetOutput( level )->SetRequestedRegionToLargestPossibleRegion();
  }
} // end GenerateOutputRequestedRegion()


/**
 * ******************* GenerateInputRequestedRegion ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::GenerateInputRequestedRegion( void )
{
  if( this->IsRescaleUsed() )
  {
    /** GenericMultiResolutionPyramidImageFilter requires a larger input requested
     * region than the output requested regions to accommodate the shrinkage and
     * smoothing operations. Therefore Superclass provides this implementation.
     */
    Superclass::GenerateInputRequestedRegion();
  }
  else
  {
    /** call the SuperSuperclass implementation of this method. This should
     * copy the output requested region to the input requested region
     */
    SuperSuperclass::GenerateInputRequestedRegion();

    /** This filter needs all of the input, because it uses the the
     * GausianRecursiveFilter.
     */
    InputImagePointer image = const_cast< InputImageType * >( this->GetInput() );

    if( !image )
    {
      itkExceptionMacro( << "Input has not been set." );
    }
    else
    {
      image->SetRequestedRegion( this->GetInput()->GetLargestPossibleRegion() );
    }
  }
} // end GenerateInputRequestedRegion()


/**
 * ******************* ReleaseOutputs ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::ReleaseOutputs( void )
{
  // release the memories if already has been allocated
  for( unsigned int level = 0; level < this->m_NumberOfLevels; level++ )
  {
    if( this->m_ComputeOnlyForCurrentLevel && level != this->m_CurrentLevel )
    {
      this->GetOutput( level )->Initialize();
    }
  }
} // end ReleaseOutputs()


/**
 * ******************* ComputeForCurrentLevel ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
bool
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::ComputeForCurrentLevel( const unsigned int level ) const
{
  if( !this->m_ComputeOnlyForCurrentLevel
    || ( this->m_ComputeOnlyForCurrentLevel && level == this->m_CurrentLevel ) )
  {
    return true;
  }
  else
  {
    return false;
  }
} // end ComputeOnlyForCurrentLevel()


/**
 * ******************* GetDefaultSigma ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
double
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::GetDefaultSigma( const unsigned int level, const unsigned int dim,
  const unsigned int * factors,
  const SpacingType & spacing ) const
{
  /** Compute the standard deviation: 0.5 * factor * spacing
   * This is exactly like in the Superclass.
   * In the superclass, the DiscreteGaussianImageFilter is used, which
   * requires the variance, and has the option to ignore the image spacing.
   * That's why the formula looks maybe different at first sight.
   */
  if( factors[ dim ] == 1 && ( level == this->m_NumberOfLevels - 1 ) ) { return 0.0; }
  return 0.5 * static_cast< double >( factors[ dim ] ) * spacing[ dim ];
} // end GetDefaultSigma()


/**
 * ******************* SetSmoothingScheduleToDefault ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::SetSmoothingScheduleToDefault( void )
{
  InputImageConstPointer input   = this->GetInput();
  const SpacingType &    spacing = input->GetSpacing();

  // Resize the smoothing schedule
  SmoothingScheduleType temp( this->GetNumberOfLevels(), ImageDimension );
  temp.Fill( 0 );
  this->m_SmoothingSchedule = temp;

  unsigned int factors[ ImageDimension ];
  for( unsigned int level = 0; level < this->m_NumberOfLevels; level++ )
  {
    for( unsigned int dim = 0; dim < ImageDimension; dim++ )
    {
      factors[ dim ]                            = this->m_Schedule[ level ][ dim ];
      this->m_SmoothingSchedule[ level ][ dim ] = this->GetDefaultSigma( level, dim, factors, spacing );
    }
  }
} // end SetSmoothingScheduleToDefault()


/**
 * ******************* GetSigma ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::GetSigma( const unsigned int level, SigmaArrayType & sigmaArray ) const
{
  sigmaArray.Fill( 0 );
  for( unsigned int dim = 0; dim < ImageDimension; dim++ )
  {
    sigmaArray[ dim ] = this->m_SmoothingSchedule[ level ][ dim ];
  }
} // end GetSigma()


/**
 * ******************* GetShrinkFactors ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::GetShrinkFactors( const unsigned int level, RescaleFactorArrayType & shrinkFactors ) const
{
  shrinkFactors.Fill( 0 );
  for( unsigned int dim = 0; dim < ImageDimension; dim++ )
  {
    /** Here we would prefer to use m_RescaleSchedule.
     * Although it would require copying most of the methods
     * from MultiResolutionPyramidImageFilter and changing m_Schedule
     * to m_RescaleSchedule.
     */
    shrinkFactors[ dim ] = this->m_Schedule[ level ][ dim ];
  }
} // end GetShrinkFactors()


/**
 * ******************* AreSigmasAllZeros ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
bool
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::AreSigmasAllZeros( const SigmaArrayType & sigmaArray ) const
{
  const ScalarRealType zero = NumericTraits< ScalarRealType >::Zero;
  for( unsigned int dim = 0; dim < ImageDimension; dim++ )
  {
    if( sigmaArray[ dim ] != zero )
    {
      return false;
    }
  }

  return true;
} // end AreSigmasAllZeros()


/**
 * ******************* AreRescaleFactorsAllOnes ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
bool
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::AreRescaleFactorsAllOnes( const RescaleFactorArrayType & rescaleFactors ) const
{
  const ScalarRealType one = NumericTraits< ScalarRealType >::One;
  for( unsigned int dim = 0; dim < ImageDimension; dim++ )
  {
    if( rescaleFactors[ dim ] != one )
    {
      return false;
    }
  }

  return true;
} // end AreRescaleFactorsAllOnes()


/**
 * ******************* IsSmoothingUsed ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
bool
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::IsSmoothingUsed( void ) const
{
  // If for any level all sigma elements are not zeros then smooth are used in pipeline
  SigmaArrayType sigmaArray;
  for( unsigned int level = 0; level < this->m_NumberOfLevels; level++ )
  {
    this->GetSigma( level, sigmaArray );
    if( !this->AreSigmasAllZeros( sigmaArray ) )
    {
      return true;
    }
  }
  return false;
} // end IsSmoothingUsed()


/**
 * ******************* IsRescaleUsed ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
bool
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::IsRescaleUsed( void ) const
{
// If for any level all rescale factors are not ones then rescale are used in pipeline
  RescaleFactorArrayType rescaleFactors;
  for( unsigned int level = 0; level < this->m_NumberOfLevels; level++ )
  {
    this->GetShrinkFactors( level, rescaleFactors );
    if( !this->AreRescaleFactorsAllOnes( rescaleFactors ) )
    {
      return true;
    }
  }
  return false;
} // end IsRescaleUsed()


/**
 * ******************* PrintSelf ***********************
 */

template< class TInputImage, class TOutputImage, class TPrecisionType >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage, TPrecisionType >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "CurrentLevel: "
     << this->m_CurrentLevel << std::endl;
  os << indent << "ComputeOnlyForCurrentLevel: "
     << ( this->m_ComputeOnlyForCurrentLevel ? "true" : "false" ) << std::endl;
  os << indent << "SmoothingScheduleDefined: "
     << ( this->m_SmoothingScheduleDefined ? "true" : "false" ) << std::endl;
  os << indent << "Smoothing Schedule: ";
  if( this->m_SmoothingSchedule.size() == 0 )
  {
    os << "Not set" << std::endl;
  }
  else
  {
    os << std::endl << this->m_SmoothingSchedule << std::endl;
  }
} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef __itkGenericMultiResolutionPyramidImageFilter_hxx
