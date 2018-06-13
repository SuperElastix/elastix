/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGenericMultiResolutionPyramidImageFilter_hxx
#define __itkGenericMultiResolutionPyramidImageFilter_hxx

#include "itkGenericMultiResolutionPyramidImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkCastImageFilter.h"

namespace // anonymous namespace
{

/**
 * ******************* Graft ***********************
 * Helper function
 */

template< class GenericMultiResolutionPyramidImageFilterType,
class ImageToImageFilterType, typename OutputImageType >
void
Graft(
  typename GenericMultiResolutionPyramidImageFilterType::Pointer thisFilter,
  typename ImageToImageFilterType::Pointer & filter,
  OutputImageType * outImage, const unsigned int ilevel )
{
  filter->GraftOutput( outImage );

  // force to always update in case shrink factors are the same
  filter->Modified();
  filter->UpdateLargestPossibleRegion();
  thisFilter->GraftNthOutput( ilevel, filter->GetOutput() );
} // end Graft()


} // end namespace anonymous

namespace itk
{

/**
 * ******************* Constructor ***********************
 */

template< class TInputImage, class TOutputImage >
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::GenericMultiResolutionPyramidImageFilter()
{
  this->m_CurrentLevel                        = 0;
  this->m_ComputeOnlyForCurrentLevel          = false;
  this->m_UseMultiResolutionRescaleSchedule   = true; // Default behavior of MultiResolutionPyramidImageFilter
  this->m_UseMultiResolutionSmoothingSchedule = true; // Default behavior of MultiResolutionPyramidImageFilter

  SmoothingScheduleType temp( this->GetNumberOfLevels(), ImageDimension );
  temp.Fill( 0 );
  this->m_SmoothingSchedule        = temp;
  this->m_SmoothingScheduleDefined = false;

} // end Constructor


/**
 * ******************* SetNumberOfLevels ***********************
 */

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::SetNumberOfLevels( unsigned int num )
{
  if( this->m_NumberOfLevels == num ) { return; }
  Superclass::SetNumberOfLevels( num );

  /** Resize the smoothing schedule too. */
  SmoothingScheduleType temp( this->m_NumberOfLevels, ImageDimension );
  temp.Fill( 0.0 );
  this->m_SmoothingSchedule        = temp;
  this->m_SmoothingScheduleDefined = false;

} // end SetNumberOfLevels()


/**
 * ******************* SetCurrentLevel ***********************
 */

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::SetCurrentLevel( unsigned int level )
{
  itkDebugMacro( "setting CurrentLevel to " << level );
  if( this->m_CurrentLevel != level )
  {
    // clamp value to be less then number of levels
    this->m_CurrentLevel = level;
    if( this->m_CurrentLevel >= this->m_NumberOfLevels )
    {
      this->m_CurrentLevel = this->m_NumberOfLevels - 1; // Safe this->m_NumberOfLevels always >= 1
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

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
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

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
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

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
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
 * ******************* SetSmoothingSchedule ***********************
 */

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
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
 * ******************* GenerateData ***********************
 */

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
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
  //    Then pipeline is: input -> caster -> shrinker/resample -> output
  // 3. m_UseMultiResolutionSmoothingSchedule = true
  //    m_UseMultiResolutionRescaleSchedule = false
  //    Then pipeline is: input -> smoother -> output
  // 4. m_UseMultiResolutionSmoothingSchedule = false
  //    m_UseMultiResolutionRescaleSchedule = false
  //    Then pipeline is: input -> caster -> output
  //
  // But for example the smoother can be skipped if AreSigmasAllZeros(...)
  // returns true for the current level. Then pipeline 1 transforms to:
  // input -> caster -> shrinker/resample -> output for the computing level.
  //
  // Pipeline also takes care of memory allocation for N'th output if
  // SetComputeOnlyForCurrentLevel has been set to true.

  // Get the input and output pointers
  InputImageConstPointer input = this->GetInput();

  // Typedef for caster
  typedef CastImageFilter< InputImageType, OutputImageType > CasterType;

  // Check if we have to do anything at all
  if( !this->m_UseMultiResolutionRescaleSchedule
    && !this->m_UseMultiResolutionSmoothingSchedule )
  {
    // Create caster
    typename CasterType::Pointer caster = CasterType::New();
    caster->SetInput( input );

    // This is a special case we just allocate output images and copy input
    for( unsigned int level = 0; level < this->m_NumberOfLevels; level++ )
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

        Graft< Self, CasterType, OutputImageType >(
          this, caster, outputPtr, level );
      }
    }
    return; // We are done, return
  }

  // Create caster, smoother, resample, and shrinker filters
  typedef SmoothingRecursiveGaussianImageFilter<
    InputImageType, OutputImageType >                            SmootherType;
  typedef ImageToImageFilter< OutputImageType, OutputImageType >  ImageToImageType;
  typedef ResampleImageFilter< OutputImageType, OutputImageType > ResampleShrinkerType;
  typedef ShrinkImageFilter< OutputImageType, OutputImageType >   ShrinkerType;

  // First check if smoothing schedule has been set
  if( !this->m_SmoothingScheduleDefined )
  {
    this->SetSmoothingScheduleToDefault();
  }

  typename CasterType::Pointer caster;
  typename SmootherType::Pointer smoother;

  if( this->m_UseMultiResolutionSmoothingSchedule )
  {
    smoother = SmootherType::New();
    smoother->SetInput( input );
    if( this->IsCasterNeeded() )
    {
      caster = CasterType::New();
      caster->SetInput( input );
    }
  }
  else
  {
    caster = CasterType::New();
    caster->SetInput( input );
  }

  /** Only one of these pointers is going to be valid, depending on the
   * value of UseShrinkImageFilter flag.
   */
  typename ImageToImageType::Pointer shrinkerFilter;
  typename ResampleShrinkerType::Pointer resampleShrinker;
  typename ShrinkerType::Pointer shrinker;

  if( this->m_UseMultiResolutionRescaleSchedule )
  {
    if( this->GetUseShrinkImageFilter() )
    {
      shrinker       = ShrinkerType::New();
      shrinkerFilter = shrinker.GetPointer();
    }
    else
    {
      resampleShrinker = ResampleShrinkerType::New();
      typedef itk::LinearInterpolateImageFunction< OutputImageType, double >
        LinearInterpolatorType;
      typename LinearInterpolatorType::Pointer interpolator
        = LinearInterpolatorType::New();
      resampleShrinker->SetInterpolator( interpolator );
      resampleShrinker->SetDefaultPixelValue( 0 );
      shrinkerFilter = resampleShrinker.GetPointer();
    }

    if( this->m_UseMultiResolutionSmoothingSchedule )
    {
      shrinkerFilter->SetInput( smoother->GetOutput() );
    }
    else
    {
      shrinkerFilter->SetInput( caster->GetOutput() );
    }
  }

  // Set the standard deviation and do the smoothing
  unsigned int   shrinkFactors[ ImageDimension ];
  SigmaArrayType sigmaArray; sigmaArray.Fill( 0 );

  for( unsigned int level = 0; level < this->m_NumberOfLevels; level++ )
  {
    if( !this->m_ComputeOnlyForCurrentLevel )
    {
      this->UpdateProgress( static_cast< float >( level )
        / static_cast< float >( this->m_NumberOfLevels ) );
    }

    OutputImagePointer outputPtr;
    if( this->ComputeForCurrentLevel( level ) )
    {
      // Allocate memory for each output
      outputPtr = this->GetOutput( level );
      outputPtr->SetBufferedRegion( outputPtr->GetRequestedRegion() );
      outputPtr->Allocate();
    }

    // compute shrink factors
    if( this->m_UseMultiResolutionRescaleSchedule )
    {
      for( unsigned int dim = 0; dim < ImageDimension; dim++ )
      {
        /** Here we would prefer to use m_RescaleSchedule.
         * Although it would require copying most of the methods
         * from MultiResolutionPyramidImageFilter and changing m_Schedule
         * to m_RescaleSchedule.
         */
        shrinkFactors[ dim ] = this->m_Schedule[ level ][ dim ];
      }
    }

    if( this->m_UseMultiResolutionSmoothingSchedule )
    {
      this->GetSigma( level, sigmaArray );
      smoother->SetSigmaArray( sigmaArray );
    }

    if( this->ComputeForCurrentLevel( level ) )
    {
      if( this->m_UseMultiResolutionRescaleSchedule )
      {
        if( !this->GetUseShrinkImageFilter() )
        {
          typedef itk::IdentityTransform< double, OutputImageType::ImageDimension >
            IdentityTransformType;
          typename IdentityTransformType::Pointer identityTransform
            = IdentityTransformType::New();
          resampleShrinker->SetOutputParametersFromImage( outputPtr );
          resampleShrinker->SetTransform( identityTransform );
        }
        else
        {
          shrinker->SetShrinkFactors( shrinkFactors );
        }

        // Swap input, if sigma array has been set to zeros we don't perform smoothing
        if( this->AreSigmasAllZeros( sigmaArray ) )
        {
          shrinkerFilter->SetInput( caster->GetOutput() );
        }
        else
        {
          shrinkerFilter->SetInput( smoother->GetOutput() );
        }

        Graft< Self, ImageToImageType, OutputImageType >(
          this, shrinkerFilter, outputPtr, level );
      }
      else
      {
        // Swap input, if sigma array has been set to zeros we don't perform smoothing
        if( this->AreSigmasAllZeros( sigmaArray ) )
        {
          Graft< Self, CasterType, OutputImageType >(
            this, caster, outputPtr, level );
        }
        else
        {
          Graft< Self, SmootherType, OutputImageType >(
            this, smoother, outputPtr, level );
        }
      }
    }
  } // end for ilevel

} // end GenerateData()


/**
 * ******************* GenerateOutputInformation ***********************
 */

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::GenerateOutputInformation( void )
{
  if( this->m_UseMultiResolutionRescaleSchedule )
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

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::GenerateOutputRequestedRegion( DataObject * refOutput )
{
  if( this->m_UseMultiResolutionRescaleSchedule )
  {
    Superclass::GenerateOutputRequestedRegion( refOutput );
  }
  else
  {
    // call the supersuperclass's implementation of this method
    SuperSuperclass::GenerateOutputRequestedRegion( refOutput );

    // DS: I don't get this part at all

    // find the index for this output
    //unsigned int refLevel = refOutput->GetSourceOutputIndex();

    // \todo: shouldn't this be a dynamic_cast?
    //TOutputImage * ptr = static_cast<TOutputImage*>( refOutput );
    //if( !ptr )
    //{
    //  itkExceptionMacro( << "Could not cast refOutput to TOutputImage*." );
    //}

    //unsigned int ilevel;
    //if ( ptr->GetRequestedRegion() == ptr->GetLargestPossibleRegion() )
    //{
    //   set the requested regions for the other outputs to their
    //   requested region
    //  for( ilevel = 0; ilevel < this->m_NumberOfLevels; ilevel++ )
    //  {
    //    if( ilevel == refLevel ) { continue; }
    //    if( !this->GetOutput(ilevel) ) { continue; }

    //    this->GetOutput(ilevel)->SetRequestedRegionToLargestPossibleRegion();
    //  }
    //}
    //else
    //{
    //   compute requested regions for the other outputs based on
    //   the requested region of the reference output

    //   Set them all to the same region
    //  typedef typename OutputImageType::RegionType RegionType;
    //  RegionType outputRegion = ptr->GetRequestedRegion();

    //  for( ilevel = 0; ilevel < this->m_NumberOfLevels; ilevel++ )
    //  {
    //    if( ilevel == refLevel ) { continue; }
    //    if( !this->GetOutput(ilevel) ) { continue; }

    //     make sure the region is within the largest possible region
    //    outputRegion.Crop( this->GetOutput( ilevel )->GetLargestPossibleRegion() );
    //     set the requested region
    //    this->GetOutput( ilevel )->SetRequestedRegion( outputRegion );
    //  }
    //}

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

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion( void )
{
  if( this->m_UseMultiResolutionRescaleSchedule )
  {
    // GenericMultiResolutionPyramidImageFilter requires a larger input requested
    // region than the output requested regions to accommodate the shrinkage and
    // smoothing operations. Therefore Superclass provides this implementation.
    Superclass::GenerateInputRequestedRegion();
  }
  else
  {
    // call the SuperSuperclass implementation of this method. This should
    // copy the output requested region to the input requested region
    SuperSuperclass::GenerateInputRequestedRegion();

    // This filter needs all of the input, because it uses the the GausianRecursiveFilter.
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

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
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

template< class TInputImage, class TOutputImage >
bool
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
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

template< class TInputImage, class TOutputImage >
double
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::GetDefaultSigma( const unsigned int dim,
  const unsigned int * factors,
  const SpacingType & spacing ) const
{
  /** Compute the standard deviation: 0.5 * factor * spacing
   * This is exactly like in the Superclass.
   * In the superclass, the DiscreteGaussianImageFilter is used, which
   * requires the variance, and has the option to ignore the image spacing.
   * That's why the formula looks maybe different at first sight.
   */
  if( factors[ dim ] == 1 ) { return 0.0; }
  return 0.5 * static_cast< double >( factors[ dim ] ) * spacing[ dim ];

} // end GetDefaultSigma()


/**
 * ******************* SetSmoothingScheduleToDefault ***********************
 */

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
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
      this->m_SmoothingSchedule[ level ][ dim ] = this->GetDefaultSigma( dim, factors, spacing );
    }
  }
} // end SetSmoothingScheduleToDefault()


/**
 * ******************* GetSigma ***********************
 */

template< class TInputImage, class TOutputImage >
typename GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >::SigmaArrayType
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::GetSigma( const unsigned int level, SigmaArrayType & sigmaArray ) const
{
  for( unsigned int dim = 0; dim < ImageDimension; dim++ )
  {
    sigmaArray[ dim ] = this->m_SmoothingSchedule[ level ][ dim ];
  }
  return sigmaArray;

} // end GetSigma()


/**
 * ******************* AreSigmasAllZeros ***********************
 */

template< class TInputImage, class TOutputImage >
bool
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::AreSigmasAllZeros( const SigmaArrayType & sigmaArray ) const
{
  if( !this->m_UseMultiResolutionSmoothingSchedule )
  {
    return true;
  }

  bool allZeros = ( sigmaArray[ 0 ] == 0.0 );
  for( unsigned int dim = 1; dim < ImageDimension; dim++ )
  {
    if( sigmaArray[ dim ] == 0.0 )
    {
      allZeros |= true;
    }
    else
    {
      allZeros |= false;
    }
  }

  return allZeros;

} // end AreSigmasAllZeros()


/**
 * ******************* AreRescaleFactorsAllOnes ***********************
 */
// \todo : USE THIS FUNCTION!

template< class TInputImage, class TOutputImage >
bool
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::AreRescaleFactorsAllOnes( const RescaleFactorArrayType & rescaleFactorArray ) const
{
  if( !this->m_UseMultiResolutionSmoothingSchedule )
  {
    return true;
  }

  bool allOnes = ( rescaleFactorArray[ 0 ] == 1.0 );
  for( unsigned int dim = 1; dim < ImageDimension; dim++ )
  {
    if( rescaleFactorArray[ dim ] == 1.0 )
    {
      allOnes |= true;
    }
    else
    {
      allOnes |= false;
    }
  }

  return allOnes;

} // end AreRescaleFactorsAllOnes()


/**
 * ******************* IsCasterNeeded ***********************
 */

template< class TInputImage, class TOutputImage >
bool
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::IsCasterNeeded( void ) const
{
  // If for any level all sigma elements are zeros then we need caster in the
  // pipeline
  bool           need = false;
  SigmaArrayType sigmaArray;
  for( unsigned int level = 0; level < this->m_NumberOfLevels; level++ )
  {
    this->GetSigma( level, sigmaArray );
    if( this->AreSigmasAllZeros( sigmaArray ) )
    {
      return true;
    }
  }
  return need;

} // end IsCasterNeeded()


/**
 * ******************* PrintSelf ***********************
 */

template< class TInputImage, class TOutputImage >
void
GenericMultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "CurrentLevel: "
     << this->m_CurrentLevel << std::endl;
  os << indent << "ComputeOnlyForCurrentLevel: "
     << ( this->m_ComputeOnlyForCurrentLevel ? "true" : "false" ) << std::endl;
  os << indent << "UseMultiResolutionRescaleSchedule: "
     << ( this->m_UseMultiResolutionRescaleSchedule ? "true" : "false" ) << std::endl;
  os << indent << "UseMultiResolutionSmoothingSchedule: "
     << ( this->m_UseMultiResolutionSmoothingSchedule ? "true" : "false" ) << std::endl;
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
