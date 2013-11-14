/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxBSplineTransformWithDiffusion_HXX__
#define __elxBSplineTransformWithDiffusion_HXX__

#include "elxBSplineTransformWithDiffusion.h"

#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineDecompositionImageFilter.h"

#include "vnl/vnl_math.h"

namespace elastix
{


/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
BSplineTransformWithDiffusion<TElastix>
::BSplineTransformWithDiffusion()
{
  /** Set up CombinationTransform */
  this->m_BSplineTransform = BSplineTransformType::New();
  this->SetCurrentTransform( this->m_BSplineTransform );

  /** Initialize some things. */
  this->m_GridSpacingFactor.Fill( 8.0 );
  this->m_Interpolator  = InterpolatorType::New();

  /** Initialize things for diffusion. */
  this->m_Diffusion = 0;
  this->m_DeformationField = 0;
  this->m_DiffusedField = 0;
  this->m_GrayValueImage1 = 0;
  this->m_GrayValueImage2 = 0;
  this->m_MovingSegmentationImage = 0;
  this->m_FixedSegmentationImage = 0;
  this->m_MovingSegmentationReader = 0;
  this->m_FixedSegmentationReader = 0;
  this->m_MovingSegmentationFileName = "";
  this->m_FixedSegmentationFileName = "";
  this->m_Resampler1 = 0;
  this->m_Resampler2 = 0;
  this->m_WriteDiffusionFiles = false;
  this->m_AlsoFixed = true;
  this->m_ThresholdBool = true;
  this->m_ThresholdHU = static_cast<GrayValuePixelType>( 150 );
  this->m_UseMovingSegmentation = false;
  this->m_UseFixedSegmentation = false;

  /** Make sure that the TransformBase::WriteToFile() does
   * not write the transformParameters in the file.
   */
  this->SetReadWriteTransformParameters( false );

} // end Constructor


/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
BSplineTransformWithDiffusion<TElastix>
::BeforeRegistration( void )
{
  /** Set initial transform parameters to a 1x1x1 grid, with deformation (0,0,0).
   * In the method BeforeEachResolution() this will be replaced by the right grid size.
   *
   * This seems not logical, but it is required, since the registration
   * class checks if the number of parameters in the transform is equal to
   * the number of parameters in the registration class. This check is done
   * before calling the BeforeEachResolution() methods.
   */

  /** Task 1 - Set the Grid. */

  /** Declarations. */
  RegionType gridregion;
  SizeType gridsize;
  IndexType gridindex;
  SpacingType gridspacing;
  OriginType gridorigin;

  /** Fill everything with default values. */
  gridsize.Fill( 1 );
  gridindex.Fill( 0 );
  gridspacing.Fill( 1.0 );
  gridorigin.Fill( 0.0 );

  /** Set it all. */
  gridregion.SetIndex( gridindex );
  gridregion.SetSize( gridsize );
  this->m_BSplineTransform->SetGridRegion( gridregion );
  this->m_BSplineTransform->SetGridSpacing( gridspacing );
  this->m_BSplineTransform->SetGridOrigin( gridorigin );

  /** Task 2 - Give the registration an initial parameter-array. */
  ParametersType dummyInitialParameters( this->GetNumberOfParameters() );
  dummyInitialParameters.Fill( 0.0 );

  /** Put parameters in the registration. */
  this->m_Registration->GetAsITKBaseType()
    ->SetInitialTransformParameters( dummyInitialParameters );

  /** Task 3: This registration uses a diffusion of the deformation field
   * every n-th iteration; the diffusion filter must be created.
   * Also allocate this->m_DeformationField and this->m_DiffusedField.
   */

  /** Get diffusion information: radius. */
  unsigned int radius1D = 1;
  this->m_Configuration->ReadParameter( radius1D, "Radius", 0 );
  RadiusType radius;
  for ( unsigned int i = 0; i < this->FixedImageDimension; i++ )
  {
    radius[ i ] = static_cast<long unsigned int>( radius1D );
  }

  /** Get diffusion information: Number of iterations. */
  unsigned int iterations = 1;
  this->m_Configuration->ReadParameter( iterations, "NumberOfDiffusionIterations", 0 );
  if ( iterations < 1 )
  {
    xout["warning"] << "WARNING: NumberOfDiffusionIterations == 0" << std::endl;
  }

  /** Get diffusion information: threshold information. */
  std::string thresholdbooltmp = "true";
  this->m_Configuration->ReadParameter( thresholdbooltmp, "ThresholdBool", 0 );
  if ( thresholdbooltmp == "false" ) this->m_ThresholdBool = false;

  float tempThresholdHU = this->m_ThresholdHU;
  this->m_Configuration->ReadParameter( tempThresholdHU, "ThresholdHU", 0 );
  this->m_ThresholdHU = static_cast< GrayValuePixelType >( tempThresholdHU );

  /** Get diffusion information: is it wanted to also take the
   * fixed image into account for the derivation of the GrayValueImage.
   */
  std::string alsoFixed = "true";
  this->m_Configuration->ReadParameter( alsoFixed,
    "GrayValueImageAlsoBasedOnFixedImage", 0 );
  if ( alsoFixed == "false" ) this->m_AlsoFixed = false;

  /** Get diffusion information: is it wanted to base the GrayValueImage
   * on a segmentation of the moving image.
   */
  std::string useMovingSegmentation = "false";
  this->m_Configuration->ReadParameter( useMovingSegmentation,
    "UseMovingSegmentation", 0 );
  if ( useMovingSegmentation == "true" ) this->m_UseMovingSegmentation = true;

  /** Get diffusion information: in case m_UseMovingSegmentation = true,
   * get the filename.
   */
  if ( this->m_UseMovingSegmentation )
  {
    this->m_Configuration->ReadParameter( this->m_MovingSegmentationFileName,
      "MovingSegmentationFileName", 0 );
    if ( m_MovingSegmentationFileName == "" )
    {
      xout[ "error" ] << "ERROR: No MovingSegmentation filename specified." << std::endl;
      /** Create and throw an exception. */
      itkExceptionMacro( << "ERROR: No MovingSegmentation filename specified.");
    }
  }

  /** Get diffusion information: is it wanted to base the GrayValueImage
   * on a segmentation of the fixed image.
   */
  std::string useFixedSegmentation = "false";
  this->m_Configuration->ReadParameter( useFixedSegmentation,
    "UseFixedSegmentation", 0 );
  if ( useFixedSegmentation == "true" ) this->m_UseFixedSegmentation = true;

  /** Get diffusion information: in case m_UseFixedSegmentation = true,
   * get the filename.
   */
  if ( this->m_UseFixedSegmentation )
  {
    this->m_Configuration->ReadParameter( this->m_FixedSegmentationFileName,
      "FixedSegmentationFileName", 0 );
    if ( m_FixedSegmentationFileName == "" )
    {
      xout[ "error" ] << "ERROR: No FixedSegmentation filename specified." << std::endl;
      /** Create and throw an exception. */
      itkExceptionMacro( << "ERROR: No FixedSegmentation filename specified." );
    }
  }

  /** Get diffusion information: Find out if the user wants
   * to write the diffusion files:
   * deformationField, GrayvalueImage, diffusedField.
   */
  std::string writetofile = "false";
  this->m_Configuration->ReadParameter( writetofile, "WriteDiffusionFiles", 0 );
  if ( writetofile == "true" ) this->m_WriteDiffusionFiles = true;

  /** Get the appropriate image information. */
  this->m_DeformationOrigin = this->m_Elastix->GetElxResamplerBase()
    ->GetAsITKBaseType()->GetOutputOrigin();
  this->m_DeformationSpacing = this->m_Elastix->GetElxResamplerBase()
    ->GetAsITKBaseType()->GetOutputSpacing();
  this->m_DeformationRegion.SetIndex( this->m_Elastix->GetElxResamplerBase()
    ->GetAsITKBaseType()->GetOutputStartIndex() );
  this->m_DeformationRegion.SetSize( this->m_Elastix->GetElxResamplerBase()
    ->GetAsITKBaseType()->GetSize() );

  /** Set it in the DeformationFieldRegulizer class. */
  this->SetDeformationFieldRegion( this->m_DeformationRegion );
  this->SetDeformationFieldOrigin( this->m_DeformationOrigin );
  this->SetDeformationFieldSpacing( this->m_DeformationSpacing );

  /** Initialize the this->m_IntermediaryDeformationFieldTransform,
   * which is in the DeformationFieldRegulizer class.
   */
  this->InitializeDeformationFields();

  /** Create this->m_DeformationField and allocate memory. */
  this->m_DeformationField = VectorImageType::New();
  this->m_DeformationField->SetRegions( this->m_DeformationRegion );
  this->m_DeformationField->SetOrigin( this->m_DeformationOrigin );
  this->m_DeformationField->SetSpacing( this->m_DeformationSpacing );
  this->m_DeformationField->Allocate();

  /** Create this->m_DeformationField and allocate memory. */
  this->m_DiffusedField = VectorImageType::New();
  this->m_DiffusedField->SetRegions( this->m_DeformationRegion );
  this->m_DiffusedField->SetOrigin( this->m_DeformationOrigin );
  this->m_DiffusedField->SetSpacing( this->m_DeformationSpacing );
  this->m_DiffusedField->Allocate();

  /** Create the GrayValueImages and allocate memory. */
  if ( this->m_UseMovingSegmentation && !this->m_ThresholdBool )
  {
    /** In this case we have to read in the m_MovingSegmentationImage. */
    this->m_MovingSegmentationReader = GrayValueImageReaderType::New();
    this->m_MovingSegmentationReader->SetFileName( m_MovingSegmentationFileName.c_str() );
    this->m_MovingSegmentationImage = m_MovingSegmentationReader->GetOutput();

    /** Read the MovingSegmentation. */
    try
    {
      this->m_MovingSegmentationImage->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      /** Add information to the exception. */
      excp.SetLocation( "BSplineTransformWithDiffusion - BeforeRegistration()" );
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while reading the MovingSegmentationImage.\n";
      excp.SetDescription( err_str );
      /** Pass the exception to an higher level. */
      throw excp;
    }

    /** In this case: check if a FixedSegmentationImage is needed. */
    if ( this->m_UseFixedSegmentation )
    {
      /** In this case we have to read in the m_FixedSegmentationImage. */
      this->m_FixedSegmentationReader = GrayValueImageReaderType::New();
      this->m_FixedSegmentationReader->SetFileName( m_FixedSegmentationFileName.c_str() );
      this->m_FixedSegmentationImage = m_FixedSegmentationReader->GetOutput();

      /** Read the FixedSegmentation. */
      try
      {
        this->m_FixedSegmentationImage->Update();
      }
      catch( itk::ExceptionObject & excp )
      {
        /** Add information to the exception. */
        excp.SetLocation( "BSplineTransformWithDiffusion - BeforeRegistration()" );
        std::string err_str = excp.GetDescription();
        err_str += "\nError occurred while reading the FixedSegmentationImage.\n";
        excp.SetDescription( err_str );
        /** Pass the exception to an higher level. */
        throw excp;
      } // end try/catch
    } // end if fixed segmentation
  } // end if moving segmentation
  /** Otherwise defining rigid object is based on thresholding the resampled moving image. */
  else if ( !this->m_UseMovingSegmentation && this->m_ThresholdBool )
  {
    this->m_GrayValueImage1 = GrayValueImageType::New();
    this->m_GrayValueImage1->SetRegions( this->m_DeformationRegion );
    this->m_GrayValueImage1->SetOrigin( this->m_DeformationOrigin );
    this->m_GrayValueImage1->SetSpacing( this->m_DeformationSpacing );
    this->m_GrayValueImage1->Allocate();
    this->m_GrayValueImage2 = GrayValueImageType::New();
  }
  else
  {
    xout[ "error" ] << "ERROR: So what are you using for the GrayValueImage,"
      << std::endl
      << "either a threshold or a segmentation, make a choice!" << std::endl;

    /** Create and throw an exception. */
    itkExceptionMacro( << "ERROR: Difficulty determining how to create the "
      << "GrayValueImage. Check your parameter file." );
  }

  /** Set the interpolator. */
  if ( this->m_UseMovingSegmentation )
  {
    this->m_Interpolator->SetSplineOrder( 0 );
  }
  else
  {
    this->m_Interpolator->SetSplineOrder( 1 );
  }

  /** Create a resampler. */
  if ( this->m_UseMovingSegmentation )
  {
    this->m_Resampler2 = ResamplerType2::New();
    this->m_Resampler2->SetTransform( this->GetIntermediaryDeformationFieldTransform() );
    this->m_Resampler2->SetInterpolator( this->m_Interpolator ); // default = LinearInterpolateImageFunction
  }
  else
  {
    this->m_Resampler1 = ResamplerType1::New();
    this->m_Resampler1->SetTransform( this->GetIntermediaryDeformationFieldTransform() );
    this->m_Resampler1->SetInterpolator( this->m_Interpolator ); // default = LinearInterpolateImageFunction
  }

  /** What are we using for defining rigid structures? */
  if ( this->m_UseMovingSegmentation && !this->m_ThresholdBool )
  {
    this->m_Resampler2->SetInput( this->m_MovingSegmentationImage );
  }
  else
  {
    this->m_Resampler1->SetInput( dynamic_cast<MovingImageELXType *>(
      this->m_Elastix->GetMovingImage() ) );
  }

  /** Get the default pixel value. */
  float defaultPixelValueForGVI = itk::NumericTraits<float>::Zero;
  if ( this->m_UseMovingSegmentation && !this->m_ThresholdBool )
  {
    this->m_Configuration->ReadParameter( defaultPixelValueForGVI,
      "DefaultPixelValueForGVI", 0 );
    this->m_Resampler2->SetDefaultPixelValue(
      static_cast<GrayValuePixelType>( defaultPixelValueForGVI ) );
  }
  else
  {
    this->m_Configuration->ReadParameter( defaultPixelValueForGVI,
      "DefaultPixelValue", 0 );
    this->m_Resampler1->SetDefaultPixelValue(
      static_cast<GrayValuePixelType>( defaultPixelValueForGVI ) );
  }

  /** Set other stuff. */
  if ( this->m_UseMovingSegmentation )
  {
    this->m_Resampler2->SetSize( this->m_DeformationRegion.GetSize() );
    this->m_Resampler2->SetOutputStartIndex( this->m_DeformationRegion.GetIndex() );
    this->m_Resampler2->SetOutputOrigin( this->m_DeformationOrigin );
    this->m_Resampler2->SetOutputSpacing( this->m_DeformationSpacing );
  }
  else
  {
    this->m_Resampler1->SetSize( this->m_DeformationRegion.GetSize() );
    this->m_Resampler1->SetOutputStartIndex( this->m_DeformationRegion.GetIndex() );
    this->m_Resampler1->SetOutputOrigin( this->m_DeformationOrigin );
    this->m_Resampler1->SetOutputSpacing( this->m_DeformationSpacing );
  }

  /** Create this->m_Diffusion, the diffusion filter. */
  this->m_Diffusion = DiffusionFilterType::New();
  this->m_Diffusion->SetRadius( radius );
  this->m_Diffusion->SetNumberOfIterations( iterations );
  this->m_Diffusion->SetGrayValueImage( this->m_GrayValueImage1 );
  this->m_Diffusion->SetInput( this->m_DeformationField );

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
BSplineTransformWithDiffusion<TElastix>
::BeforeEachResolution( void )
{
  /** What is the current resolution level? */
  unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** What is the UpsampleGridOption?
   * This option defines the user's wish:
   * - true: For lower resolution levels (i.e. smaller images),
   *         the GridSpacing is made larger, as a power of 2.
   * - false: The GridSpacing remains equal for each resolution level.
   */
  std::string upsampleBSplineGridOption( "true" );
  bool upsampleGridOption = true;
  this->m_Configuration->ReadParameter( upsampleBSplineGridOption,
    "UpsampleGridOption", 0, false );
  if ( upsampleBSplineGridOption == "true" ) upsampleGridOption = true;
  else if ( upsampleBSplineGridOption == "false" ) upsampleGridOption = false;

  /** Resample the grid. */
  if ( level == 0 )
  {
    /** Set grid equal to lowest resolution fixed image. */
    this->SetInitialGrid( upsampleGridOption );
  }
  else
  {
    /** If wanted, we upsample the grid of control points. */
    if ( upsampleGridOption ) this->IncreaseScale();
    /** Otherwise, nothing is done with the B-spline grid. */
  }

} // end BeforeEachResolution()


/**
 * ********************* AfterEachIteration *********************
 */

template <class TElastix>
void
BSplineTransformWithDiffusion<TElastix>
::AfterEachIteration( void )
{
  /** Declare boolean. */
  bool DiffusionNow = false;

  /** Find out filter pattern. */
  unsigned int filterPattern = 1;
  this->m_Configuration->ReadParameter( filterPattern, "FilterPattern", 0 );
  if ( filterPattern != 1 && filterPattern != 2 )
  {
    filterPattern = 1;
    xout["warning"] << "WARNING: filterPattern set to 1" << std::endl;
  }

  /** Get the current iteration number. */
  unsigned int CurrentIterationNumber = this->m_Elastix->GetIterationCounter();

  /** Get the MaximumNumberOfIterations of this resolution level. */
  unsigned int ResNr = this->m_Elastix->GetElxRegistrationBase()
    ->GetAsITKBaseType()->GetCurrentLevel();
  unsigned int maximumNumberOfIterations = 0;
  this->m_Configuration->ReadParameter( maximumNumberOfIterations,
    "MaximumNumberOfIterations", ResNr );

  /** Find out if we have to filter now.
   * FilterPattern1: diffusion every n iterations
   * FilterPattern2: start with diffusion every n1 iterations,
   *    followed by diffusion every n2 iterations, and ended
   *    by by diffusion every n3 iterations.
   */
  if ( filterPattern == 1 )
  {
    /** Find out after how many iterations a diffusion is wanted. */
    unsigned int diffusionEachNIterations = 0;
    this->m_Configuration->ReadParameter( diffusionEachNIterations,
      "DiffusionEachNIterations", 0 );

    /** Checking DiffusionEachNIterations. */
    if ( diffusionEachNIterations < 1 )
    {
      xout["warning"] << "WARNING: DiffusionEachNIterations < 1" << std::endl;
      xout["warning"] << "\t\tDiffusionEachNIterations is set to 1" << std::endl;
      diffusionEachNIterations = 1;
    }

    /** Determine if diffusion is wanted after this iteration:
     * Do it every n iterations, but not at the first iteration
     * of a resolution, and also at the last iteration.
     */
    DiffusionNow = ( ( CurrentIterationNumber + 1 ) % diffusionEachNIterations == 0 );
    DiffusionNow &= ( CurrentIterationNumber != 0 );
    DiffusionNow |= ( CurrentIterationNumber == ( maximumNumberOfIterations - 1 ) );
  }
  else if ( filterPattern == 2 )
  {
    /** Find out after how many iterations a change in n_i is needed. */
    unsigned int afterIterations0 = 50;
    unsigned int afterIterations1 = 100;
    this->m_Configuration->ReadParameter( afterIterations0, "AfterIterations", 0 );
    this->m_Configuration->ReadParameter( afterIterations1, "AfterIterations", 1 );

    /** Find out n1, n2 and n3. */
    unsigned int howManyIterations0 = 1;
    unsigned int howManyIterations1 = 5;
    unsigned int howManyIterations2 = 10;
    this->m_Configuration->ReadParameter( howManyIterations0, "HowManyIterations", 0 );
    this->m_Configuration->ReadParameter( howManyIterations1, "HowManyIterations", 1 );
    this->m_Configuration->ReadParameter( howManyIterations2, "HowManyIterations", 2 );

    /** The first afterIterations0 the deformationField is filtered
     * every howManyIterations0 iterations. Then, for iterations between
     * afterIterations0 and afterIterations1 , the deformationField
     * is filtered after every howManyIterations1 iterations. Finally,
     * the deformationField is filtered every howManyIterations2 iterations.
     */
    unsigned int diffusionEachNIterations;
    if ( CurrentIterationNumber < afterIterations0 )
    {
      diffusionEachNIterations = howManyIterations0;
    }
    else if ( CurrentIterationNumber >= afterIterations0
      && CurrentIterationNumber < afterIterations1 )
    {
      diffusionEachNIterations = howManyIterations1;
    }
    else
    {
      diffusionEachNIterations = howManyIterations2;
    }

    /** Filter the current iteration? Also filter after the last iteration. */
    DiffusionNow = ( ( CurrentIterationNumber + 1 ) % diffusionEachNIterations == 0 );
    DiffusionNow |= ( CurrentIterationNumber == ( maximumNumberOfIterations - 1 ) );

  } // end if filterpattern

  /** If wanted (DiffusionNow == true), do a diffusion. */
  if ( DiffusionNow )
  {
    this->DiffuseDeformationField();
  }

} // end AfterEachIteration()


/**
 * ******************* AfterRegistration ***********************
 */

template <class TElastix>
void
BSplineTransformWithDiffusion<TElastix>
::AfterRegistration( void )
{
  /** Destruct some member variables that are not necessary to keep in
   * memory. Only those variables needed for the transform parameters
   * have to be kept.
   */
  this->m_GrayValueImage1 = 0;
  this->m_GrayValueImage2 = 0;
  this->m_Resampler1 = 0;
  this->m_Resampler2 = 0;
  this->m_MovingSegmentationReader = 0;
  this->m_FixedSegmentationReader = 0;
  this->m_MovingSegmentationImage = 0;
  this->m_FixedSegmentationImage = 0;
  this->m_Diffusion = 0;

  /** In the very last iteration of the registration in the function
   * DiffuseDeformationField() the intermediary deformation field is updated:
   * this->UpdateIntermediaryDeformationFieldTransform( this->m_DiffusedField );
   * This function copies the deformation field into number of dimensions
   * coefficient images (the DeformationFieldTransform is actually a
   * BSplineTransform).
   * Therefore the memory of the deformation fields can be freed.
   */
  this->m_DeformationField = 0;
  this->m_DiffusedField = 0;

} // end AfterRegistration()


/**
 * ********************* SetInitialGrid *************************
 *
 * Set the size of the initial control point grid.
 *
 * If multiresolution (UpsampleGridOption is "true") then the
 * grid size is equal to the size of the fixed image, divided by
 * desired final gridspacing and a factor 2^(NrOfImageResolutions-1).
 * Otherwise it's equal to the size of the fixed image, divided by
 * the desired final gridspacing.
 *
 * In both cases some extra grid points are put at the edges,
 * to take into account the support region of the B-splines.
 */

template <class TElastix>
void
BSplineTransformWithDiffusion<TElastix>
::SetInitialGrid( bool upsampleGridOption )
{
  /** Declarations. */
  RegionType  gridregion;
  SizeType    gridsize;
  IndexType   gridindex;
  SpacingType gridspacing;
  OriginType  gridorigin;

  /** Get the fixed image. */
  typename FixedImageType::Pointer fixedimage;
  fixedimage = const_cast< FixedImageType * >(
    this->m_Registration->GetAsITKBaseType()->GetFixedImage() );

  /** Get the size etc. of this image. */

  /** In elastix <=3.001: gridregion  = fixedimage->GetRequestedRegion();  */
  /** later (because requested regions were not supported anyway consistently: */
  gridregion = fixedimage->GetLargestPossibleRegion();
  /** \todo: allow the user to enter a region of interest for the registration.
   * Especially the boundary conditions have to be dealt with carefully then.
   */
  gridindex   = gridregion.GetIndex();
  /** \todo: always 0? doesn't a largest possible region have an index 0 by definition? */
  gridsize    = gridregion.GetSize();
  gridspacing = fixedimage->GetSpacing();
  gridorigin  = fixedimage->GetOrigin();

  /** Read the desired grid spacing for each dimension. If only one gridspacing factor
   * is given, that one is used for each dimension.
   */
  this->m_GridSpacingFactor[ 0 ] = 8.0;
  this->m_Configuration->ReadParameter( this->m_GridSpacingFactor[ 0 ], "FinalGridSpacing", 0 );
  this->m_GridSpacingFactor.Fill( this->m_GridSpacingFactor[ 0 ] );
  for ( unsigned int j = 1; j < SpaceDimension; j++ )
  {
    this->m_Configuration->ReadParameter( this->m_GridSpacingFactor[ j ], "FinalGridSpacing", j );
  }

  /** If multigrid, then start with a lower resolution grid. */
  if ( upsampleGridOption )
  {
    int nrOfResolutions = static_cast<int>(
      this->GetRegistration()->GetAsITKBaseType()->GetNumberOfLevels()  );
    this->m_GridSpacingFactor *= vcl_pow( 2.0,
      static_cast<double>( nrOfResolutions - 1 ) );
  }

  /** Determine the correct grid size. */
  for ( unsigned int j = 0; j < SpaceDimension; j++ )
  {
    gridspacing[ j ] = gridspacing[ j ] * this->m_GridSpacingFactor[ j ];
    gridorigin[ j ] -= gridspacing[ j ] *
      vcl_floor( static_cast<double>( SplineOrder ) / 2.0 );
    gridindex[ j ] = 0; // isn't this always the case anyway?
    gridsize[ j ]= static_cast< typename RegionType::SizeValueType >
      ( vcl_ceil( gridsize[ j ] / this->m_GridSpacingFactor[ j ] ) + SplineOrder );
  }

  /** Set the size data in the transform. */
  gridregion.SetSize( gridsize );
  gridregion.SetIndex( gridindex );
  this->m_BSplineTransform->SetGridRegion( gridregion );
  this->m_BSplineTransform->SetGridSpacing( gridspacing );
  this->m_BSplineTransform->SetGridOrigin( gridorigin );

  /** Set initial parameters to 0.0. */
  ParametersType initialParameters( this->GetNumberOfParameters() );
  initialParameters.Fill( 0.0 );
  this->m_Registration->GetAsITKBaseType()
    ->SetInitialTransformParametersOfNextLevel( initialParameters );

} // end SetInitialGrid()


/**
 * *********************** IncreaseScale ************************
 *
 * Upsample the grid of control points.
 */

template <class TElastix>
void
BSplineTransformWithDiffusion<TElastix>
::IncreaseScale( void )
{
  /** Typedefs. */
  typedef itk::ResampleImageFilter<ImageType, ImageType>
    UpsampleFilterType;
  typedef itk::IdentityTransform<CoordRepType, SpaceDimension>
    IdentityTransformType;
  typedef itk::BSplineResampleImageFunction<ImageType, CoordRepType>
    CoefficientUpsampleFunctionType;
  typedef itk::BSplineDecompositionImageFilter<ImageType,ImageType>
    DecompositionFilterType;
  typedef itk::ImageRegionConstIterator<ImageType>
    IteratorType;

  /** The current region/spacing settings of the grid. */
  RegionType gridregionLow = this->m_BSplineTransform->GetGridRegion();
  //SizeType gridsizeLow = gridregionLow.GetSize();
  //IndexType gridindexLow = gridregionLow.GetIndex();
  //SpacingType gridspacingLow = this->m_BSplineTransform->GetGridSpacing();
  //OriginType gridoriginLow = this->m_BSplineTransform->GetGridOrigin();

  /** Get the fixed image. */
  typename FixedImageType::Pointer fixedimage;
  fixedimage = const_cast< FixedImageType * >(
    this->m_Registration->GetAsITKBaseType()->GetFixedImage() );

  /** Set start values for computing the new grid size. */
  RegionType gridregionHigh = fixedimage->GetLargestPossibleRegion();
  IndexType gridindexHigh   = gridregionHigh.GetIndex();
  SizeType gridsizeHigh   = gridregionHigh.GetSize();
  SpacingType gridspacingHigh = fixedimage->GetSpacing();
  OriginType gridoriginHigh = fixedimage->GetOrigin();

  /** A twice as dense grid: */
  this->m_GridSpacingFactor /= 2;

  /** Determine the correct grid size. */
  for ( unsigned int j = 0; j < SpaceDimension; j++ )
  {
    gridspacingHigh[ j ] = gridspacingHigh[ j ] * this->m_GridSpacingFactor[ j ];
    gridoriginHigh[ j ] -= gridspacingHigh[ j ] *
      vcl_floor( static_cast<double>( SplineOrder ) / 2.0 );
    gridindexHigh[ j ] = 0; // isn't this always the case anyway?
    gridsizeHigh[j]= static_cast< typename RegionType::SizeValueType >
      ( vcl_ceil( gridsizeHigh[ j ] / this->m_GridSpacingFactor[ j ] ) + SplineOrder );
  }
  gridregionHigh.SetSize( gridsizeHigh );
  gridregionHigh.SetIndex( gridindexHigh );

  /** Get the latest transform parameters. */
  ParametersType latestParameters =
    this->m_Registration->GetAsITKBaseType()->GetLastTransformParameters();

  /** Get the pointer to the data in latestParameters. */
  PixelType * dataPointer = static_cast<PixelType *>( latestParameters.data_block() );
  /** Get the number of pixels that should go into one coefficient image. */
  unsigned int numberOfPixels =
    ( this->m_BSplineTransform->GetGridRegion() ).GetNumberOfPixels();

  /** Set the correct region/size info of the coefficient image
   * that will be filled with the current parameters.
   */
  ImagePointer coeffs1 = ImageType::New();
  coeffs1->SetRegions( this->m_BSplineTransform->GetGridRegion() );
  coeffs1->SetOrigin( (this->m_BSplineTransform->GetGridOrigin()).GetDataPointer() );
  coeffs1->SetSpacing( (this->m_BSplineTransform->GetGridSpacing()).GetDataPointer() );
  //coeffs1->Allocate() not needed because the data is set by directly pointing
  // to an existing piece of memory.

  /**
   * Create the new vector of parameters, with the
   * correct size (which is now approx 2^dim as big as the
   * size in the previous resolution!).
   */
  ParametersType parameters_out(
    gridregionHigh.GetNumberOfPixels() * SpaceDimension );

  /** Initialize iterator in the parameters_out. */
  unsigned int i = 0;

  /** Loop over dimension. */
  for ( unsigned int j = 0; j < SpaceDimension; j++ )
  {
    /** Fill the coefficient image with parameter data (displacements
     * of the control points in the direction of dimension j).
     */
    coeffs1->GetPixelContainer()->
      SetImportPointer( dataPointer, numberOfPixels );
    dataPointer += numberOfPixels;

    /*
     * Set this image as the input of the upsampler filter. The
     * upsampler samples the deformation field at the locations
     * of the new control points (note: it does not just interpolate
     * the coefficient image, which would be wrong). The B-spline
     * coefficients that describe the resulting image are computed
     * by the decomposition filter.
     *
     * This code is copied from the itk-example
     * DeformableRegistration6.cxx .
     */

    typename UpsampleFilterType::Pointer upsampler
      = UpsampleFilterType::New();
    typename IdentityTransformType::Pointer identity
      = IdentityTransformType::New();
    typename CoefficientUpsampleFunctionType::Pointer coeffUpsampleFunction
      = CoefficientUpsampleFunctionType::New();
    typename DecompositionFilterType::Pointer decompositionFilter
      = DecompositionFilterType::New();

    upsampler->SetInterpolator( coeffUpsampleFunction );
    upsampler->SetTransform( identity );
    upsampler->SetSize( gridsizeHigh );
    upsampler->SetOutputStartIndex( gridindexHigh );
    upsampler->SetOutputSpacing( gridspacingHigh );
    upsampler->SetOutputOrigin( gridoriginHigh );
    upsampler->SetInput( coeffs1 );

    decompositionFilter->SetSplineOrder( SplineOrder );
    decompositionFilter->SetInput( upsampler->GetOutput() );

    /** Do the upsampling. */
    try
    {
      decompositionFilter->UpdateLargestPossibleRegion();
    }
    catch( itk::ExceptionObject & excp )
    {
      /** Add information to the exception. */
      excp.SetLocation( "BSplineTransform - IncreaseScale()" );
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while using decompositionFilter.\n";
      excp.SetDescription( err_str );
      /** Pass the exception to an higher level. */
      throw excp;
    }

    /** Create an upsampled image. */
    ImagePointer coeffs2 = decompositionFilter->GetOutput();

    /** Create an iterator on the new coefficient image. */
    IteratorType iterator( coeffs2, gridregionHigh );
    iterator.GoToBegin();
    while ( !iterator.IsAtEnd() )
    {
      /** Copy the contents of coeffs2 in a ParametersType array. */
      parameters_out[ i ] = iterator.Get();
      ++iterator;
      ++i;
    } // end while coeff2 iterator loop

  } // end for dimension loop

  /** Set the initial parameters for the next resolution level. */
  this->m_BSplineTransform->SetGridRegion( gridregionHigh );
  this->m_BSplineTransform->SetGridOrigin( gridoriginHigh );
  this->m_BSplineTransform->SetGridSpacing( gridspacingHigh );
  this->m_Registration->GetAsITKBaseType()
    ->SetInitialTransformParametersOfNextLevel( parameters_out );

}  // end IncreaseScale()


/**
 * ************************* ReadFromFile ************************
 */

template <class TElastix>
void
BSplineTransformWithDiffusion<TElastix>
::ReadFromFile( void )
{
  /** Task 1 - Get and Set the DeformationField Image. */

  /** Read the name of the deformationFieldImage. */
  std::string fileName = "";
  this->m_Configuration->ReadParameter( fileName, "DeformationFieldFileName", 0 );

  /** Error checking ... */
  if ( fileName == "" )
  {
    xout["error"] << "ERROR: DeformationFieldFileName not specified."
      << std::endl << "Unable to read and set the transform parameters." << std::endl;
    // \todo quit program nicely or throw an exception
  }

  /** Read in the deformationField image. */
  typename VectorReaderType::Pointer vectorReader = VectorReaderType::New();
  vectorReader->SetFileName( fileName.c_str() );

  /** Do the reading. */
  try
  {
    vectorReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "BSplineTransformWithDiffusion - ReadFromFile()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError while reading the deformationFieldImage.\n";
    excp.SetDescription( err_str );
    /** Pass the exception to an higher level. */
    throw excp;
  }

  /** Get image information and set it in the DeformationFieldTransform. */
  RegionType region = vectorReader->GetOutput()->GetLargestPossibleRegion();
  SpacingType spacing = vectorReader->GetOutput()->GetSpacing();
  OriginType origin = vectorReader->GetOutput()->GetOrigin();
  this->SetDeformationFieldRegion( region );
  this->SetDeformationFieldSpacing( spacing );
  this->SetDeformationFieldOrigin( origin );
  this->InitializeDeformationFields();

  /** Set the deformationFieldImage in the itkDeformationVectorFieldTransform. */
  this->UpdateIntermediaryDeformationFieldTransform( vectorReader->GetOutput() );

  /** Task 2 - Get and Set the B-spline part of this transform. */

  /** Declarations of the B-spline grid and fill everything with default values. */
  RegionType  gridregion;
  SizeType    gridsize;
  IndexType   gridindex;
  SpacingType gridspacing;
  OriginType  gridorigin;
  gridsize.Fill(1);
  gridindex.Fill(0);
  gridspacing.Fill(1.0);
  gridorigin.Fill(0.0);

  /** Get GridSize, GridIndex, GridSpacing and GridOrigin. */
  for ( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    this->m_Configuration->ReadParameter( gridsize[ i ], "GridSize", i );
    this->m_Configuration->ReadParameter( gridindex[ i ], "GridIndex", i );
    this->m_Configuration->ReadParameter( gridspacing[ i ], "GridSpacing", i );
    this->m_Configuration->ReadParameter( gridorigin[ i ], "GridOrigin", i );
  }

  /** Set the B-spline grid. */
  gridregion.SetIndex( gridindex );
  gridregion.SetSize( gridsize );
  this->m_BSplineTransform->SetGridRegion( gridregion );
  this->m_BSplineTransform->SetGridSpacing( gridspacing );
  this->m_BSplineTransform->SetGridOrigin( gridorigin );

  /** Fill and set the B-spline parameters. */
  unsigned int nop = 0;
  this->m_Configuration->ReadParameter( nop, "NumberOfParameters", 0 );
  this->m_BSplineParameters.SetSize( nop );
  this->m_BSplineParameters.Fill(0.0);
  this->SetParameters( this->m_BSplineParameters );

  /** Do not call the ReadFromFile from the TransformBase,
   * because that function tries to read parameters from the file,
   * which is not necessary in this case, because the parameters are
   * in the vectorImage.
   * NOT: this->Superclass2::ReadFromFile();
   * However, we do have to copy some of the functionality:
   */

  /** Task 3 - Get and Set the Initial Transform. */

  /** Get the InitialTransformName. */
  fileName = "";
  this->m_Configuration->ReadParameter( fileName,
    "InitialTransformParametersFileName", 0 );

  /** Call the function ReadInitialTransformFromFile.*/
  if ( fileName != "NoInitialTransform" )
  {
    this->ReadInitialTransformFromFile( fileName.c_str() );
  }

  /** Task 3 - Read from the configuration file how to combine the
   * initial transform with the current transform.
   */
  std::string howToCombineTransforms = "Add"; // default
  this->m_Configuration->ReadParameter( howToCombineTransforms,
    "HowToCombineTransforms", 0, false );

  /** Convert 'this' to a pointer to a CombinationTransformType and set how
   * to combine the current transform with the initial transform */
  /** Cast to transform grouper. */
  CombinationTransformType * thisAsGrouper =
    dynamic_cast< CombinationTransformType * >(this);
  if ( thisAsGrouper )
  {
    if (howToCombineTransforms == "Compose" )
    {
      thisAsGrouper->SetUseComposition( true );
    }
    else
    {
      thisAsGrouper->SetUseComposition( false );
    }
  }

  /** Task 4 - Remember the name of the TransformParametersFileName.
   * This will be needed when another transform will use this transform
   * as an initial transform (see the WriteToFile method)
   */
  this->SetTransformParametersFileName(
    this->GetConfiguration()->GetCommandLineArgument( "-tp" ).c_str() );

} // end ReadFromFile()


/**
 * ************************* WriteToFile ************************
 *
 * Saves the TransformParameters as a vector and if wanted
 * also as a deformation field.
 */

template <class TElastix>
void
BSplineTransformWithDiffusion<TElastix>
::WriteToFile( const ParametersType & param ) const
{
  /** Call the WriteToFile from the TransformBase.*/
  this->Superclass2::WriteToFile( param );

  /** Add some BSplineTransformWithDiffusion specific lines.*/
  xout["transpar"] << std::endl << "// BSplineTransformWithDiffusion specific" << std::endl;

  /** Get the last part of the filename of the transformParameter-file,
   * which is going to be part of the filename of the deformationField image.
   */
  std::string ctpfn = this->GetElastix()->GetCurrentTransformParameterFileName();
  std::basic_string<char>::size_type pos = ctpfn.rfind( "TransformParameters." );
  std::string lastpart = ctpfn.substr( pos + 19, ctpfn.size() - pos - 19 - 4 );

  /** Write the filename of the deformationField image. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter( resultImageFormat, "ResultImageFormat", 0, false );
  std::ostringstream makeFileName( "" );
  makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
    << "DeformationFieldImage"
    << lastpart
    << "." << resultImageFormat;
  xout["transpar"] << "(DeformationFieldFileName \""
    << makeFileName.str() << "\")" << std::endl;

  /** Write the deformation field image. */
  typename DeformationFieldWriterType::Pointer writer
    = DeformationFieldWriterType::New();
  writer->SetFileName( makeFileName.str().c_str() );
  writer->SetInput( this->m_DiffusedField );

  /** Do the writing. */
  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "BSplineTransformWithDiffusion - WriteToFile()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError while writing the deformationFieldImage.\n";
    excp.SetDescription( err_str );
    /** Print the exception. */
    xl::xout["error"] << excp << std::endl;
  }

  /** Get the GridSize, GridIndex, GridSpacing and
   * GridOrigin of this transform.
   */
  SizeType size = this->m_BSplineTransform->GetGridRegion().GetSize();
  IndexType index = this->m_BSplineTransform->GetGridRegion().GetIndex();
  SpacingType spacing = this->m_BSplineTransform->GetGridSpacing();
  OriginType origin = this->m_BSplineTransform->GetGridOrigin();

  /** Write the GridSize of this transform.*/
  xout["transpar"] << "(GridSize ";
  for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
  {
    xout["transpar"] << size[ i ] << " ";
  }
  xout["transpar"] << size[ SpaceDimension - 1 ] << ")" << std::endl;

  /** Write the GridIndex of this transform.*/
  xout["transpar"] << "(GridIndex ";
  for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
  {
    xout["transpar"] << index[ i ] << " ";
  }
  xout["transpar"] << index[ SpaceDimension - 1 ] << ")" << std::endl;

  /** Set the precision of cout to 2, because GridSpacing and
   * GridOrigin must have at least one digit precision.
   */
  xout["transpar"] << std::setprecision(10);

  /** Write the GridSpacing of this transform.*/
  xout["transpar"] << "(GridSpacing ";
  for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
  {
    xout["transpar"] << spacing[ i ] << " ";
  }
  xout["transpar"] << spacing[ SpaceDimension - 1 ] << ")" << std::endl;

  /** Write the GridOrigin of this transform.*/
  xout["transpar"] << "(GridOrigin ";
  for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
  {
    xout["transpar"] << origin[ i ] << " ";
  }
  xout["transpar"] << origin[ SpaceDimension - 1 ] << ")" << std::endl;

  /** Set the precision back to default value.*/
  xout["transpar"] << std::setprecision(
    this->m_Elastix->GetDefaultOutputPrecision() );

} // end WriteToFile()


/**
 * ******************* DiffuseDeformationField ******************
 */

template <class TElastix>
void
BSplineTransformWithDiffusion<TElastix>
::DiffuseDeformationField( void )
{
  /** This function does:
   * 1) Calculate current deformation field.
   * 2) Update the intermediary deformationFieldTransform
   *    with this deformation field.
   * 3) Calculate the GrayValueImage with the resampler,
   *    which is over the intermediary deformationFieldTransform.
   * 4) Diffuse the current deformation field.
   * 5) Update the intermediary deformationFieldTransform
   *    with this diffused deformation field.
   * 6) Reset the parameters of the BSplineTransform
   *    and the optimizer. Reset the initial transform.
   * 7) If wanted, write the deformationField, the
   *    GrayValueImage and the diffusedField.
   */

  /** ------------- 1: Create deformationField. ------------- */

  /** First, create a dummyImage with the right region info, so
   * that the TransformIndexToPhysicalPoint-functions will be right.
   */
  typename DummyImageType::Pointer dummyImage = DummyImageType::New();
  dummyImage->SetRegions( this->m_DeformationRegion );
  dummyImage->SetOrigin( this->m_DeformationOrigin );
  dummyImage->SetSpacing( this->m_DeformationSpacing );

  /** Setup an iterator over dummyImage and outputImage. */
  DummyIteratorType       iter( dummyImage, this->m_DeformationRegion );
  VectorImageIteratorType iterout( this->m_DeformationField, this->m_DeformationRegion );

  /** Declare stuff. */
  InputPointType  inputPoint;
  OutputPointType outputPoint;
  VectorType      diff_point;
  IndexType       inputIndex;

  /** Calculate the TransformPoint of all voxels of the image. */
  iter.GoToBegin();
  iterout.GoToBegin();
  while ( !iter.IsAtEnd() )
  {
    inputIndex = iter.GetIndex();
    /** Transform the points to physical space. */
    dummyImage->TransformIndexToPhysicalPoint( inputIndex, inputPoint );
    /** Call TransformPoint. */
    outputPoint = this->TransformPoint( inputPoint );
    /** Calculate the difference. */
    for ( unsigned int i = 0; i < this->FixedImageDimension; i++ )
    {
      diff_point[ i ] = outputPoint[ i ] - inputPoint[ i ];
    }
    iterout.Set( diff_point );
    ++iter;
    ++iterout;
  }

  /** ------------- 2: Update the intermediary deformationFieldTransform. ------------- */

  this->UpdateIntermediaryDeformationFieldTransform( this->m_DeformationField );

  /** ------------- 3: Create GrayValueImage. ------------- */

  /** This gives:
   * - either a deformed moving image, in case that the grayValueImage
   *   is based on a threshold and not on a segmentation,
   * - or a deformed segmentation of the moving image, in case that
   *   the grayValueImage is based on a segmentation and not on a threshold.
   */
  if ( this->m_UseMovingSegmentation )
  {
    this->m_Resampler2->Modified();
    this->m_GrayValueImage1 = this->m_Resampler2->GetOutput();
  }
  else
  {
    this->m_Resampler1->Modified();
    this->m_GrayValueImage1 = this->m_Resampler1->GetOutput();
  }

  /** Do the resampling. */
  try
  {
    this->m_GrayValueImage1->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "BSplineTransformWithDiffusion - DiffuseDeformationField()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while resampling the grayValue image.\n";
    excp.SetDescription( err_str );
    /** Pass the exception to an higher level. */
    throw excp;
  }

  /** First we make a distinction between using segmentation or not. */
  if ( !this->m_UseMovingSegmentation )
  {
    /** If wanted also take the fixed image into account
     * for the derivation of the GrayValueImage, by taking the maximum.
     */
    typename MaximumImageFilterType::Pointer maximumImageFilter;
    if( this->m_AlsoFixed )
    {
      maximumImageFilter = MaximumImageFilterType::New();
      maximumImageFilter->SetInput( 0, this->m_GrayValueImage1 );
      maximumImageFilter->SetInput( 1, dynamic_cast<FixedImageELXType *>(
        this->m_Elastix->GetFixedImage() ) );
      this->m_GrayValueImage2 = maximumImageFilter->GetOutput();

      /** Do the maximum (OR filter). */
      try
      {
        this->m_GrayValueImage2->Update();
      }
      catch( itk::ExceptionObject & excp )
      {
        /** Add information to the exception. */
        excp.SetLocation( "BSplineTransformWithDiffusion - DiffuseDeformationField()" );
        std::string err_str = excp.GetDescription();
        err_str += "\nError occurred when using the maximumImageFilter to get the grayValue image.\n";
        excp.SetDescription( err_str );
        /** Pass the exception to an higher level. */
        throw excp;
      }
    } // end if alsoFixed

    if ( this->m_ThresholdBool )
    {
      /** Setup iterator. */
      GrayValueImageIteratorType it( this->m_GrayValueImage2,
        this->m_GrayValueImage2->GetLargestPossibleRegion() );
      it.GoToBegin();
      while ( !it.IsAtEnd() )
      {
        /** Threshold or just make sure everything is between 0 and 100. */
        // \todo Possibly combine this with the rescaleIntensity filter of
        //    the vectorMeanDiffusionImageFilter, in order to speed up.
        if ( it.Get() < this->m_ThresholdHU ) it.Set( 0 );
        if ( it.Get() >= this->m_ThresholdHU ) it.Set( 100 );
        /** Update iterator. */
        ++it;
      } // end while
    } // end if
  }
  /** In case we do use a segmentation of the moving image: */
  else
  {
    typename MaximumImageFilterType::Pointer maximumImageFilter;
    /** Check if we also want to use a segmentation of the fixed image. */
    if ( this->m_UseFixedSegmentation )
    {
      maximumImageFilter = MaximumImageFilterType::New();
      maximumImageFilter->SetInput( 0, this->m_GrayValueImage1 );
      maximumImageFilter->SetInput( 1, this->m_FixedSegmentationImage );
      this->m_GrayValueImage2 = maximumImageFilter->GetOutput();

      /** Do the maximum (OR filter). */
      try
      {
        this->m_GrayValueImage2->Update();
      }
      catch( itk::ExceptionObject & excp )
      {
        /** Add information to the exception. */
        excp.SetLocation( "BSplineTransformWithDiffusion - DiffuseDeformationField()" );
        std::string err_str = excp.GetDescription();
        err_str += "\nError occurred when using the maximumImageFilter to get the grayValue image.\n";
        excp.SetDescription( err_str );
        /** Pass the exception to an higher level. */
        throw excp;
      }
    }
  }

  /** ------------- 4: Setup the diffusion. ------------- */

  if( this->m_AlsoFixed || this->m_UseFixedSegmentation )
  {
    this->m_Diffusion->SetGrayValueImage( this->m_GrayValueImage2 );
  }
  else
  {
    this->m_Diffusion->SetGrayValueImage( this->m_GrayValueImage1 );
  }
  this->m_Diffusion->SetInput( this->m_DeformationField );
  this->m_Diffusion->Modified();
  this->m_DiffusedField = this->m_Diffusion->GetOutput();

  /** Diffuse deformationField. */
  try
  {
    this->m_DiffusedField->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "BSplineTransformWithDiffusion - DiffuseDeformationField()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while diffusing the deformation field.\n";
    excp.SetDescription( err_str );
    /** Pass the exception to an higher level. */
    throw excp;
  }

  /** ------------- 5: Update the intermediary transform. ------------- */

  this->UpdateIntermediaryDeformationFieldTransform( this->m_DiffusedField );

  /** ------------- 6: Reset the current transform parameters of the optimizer. ------------- */

  /** Create a vector of zeros in order to reset the B-spline transform. */
  ParametersType dummyParameters( this->GetNumberOfParameters() );
  dummyParameters.Fill( 0.0 );
  this->SetParameters( dummyParameters );

  /** Reset the optimizer.
   * We had to create the SetCurrentPositionPublic-function, because
   * SetCurrentPosition() is protected.
   */
  this->m_Elastix->GetElxOptimizerBase()->SetCurrentPositionPublic( dummyParameters );

  /** Get rid of the initial transform, because this is now captured
   * within the DeformationFieldTransform.
   */
  this->Superclass2::SetInitialTransform( 0 );

  /** ------------- 7: Write images. ------------- */

  /** If wanted, write the deformationField, the GrayValueImage and the diffusedField. */
  if ( this->m_WriteDiffusionFiles )
  {
    /** Create parts of the filenames. */
    std::string resultImageFormat = "mhd";
    this->m_Configuration->ReadParameter( resultImageFormat, "ResultImageFormat", 0, false );
    std::ostringstream makeFileName1( "" ), begin(""), end("");
    begin << this->m_Configuration->GetCommandLineArgument( "-out" );
    end << ".R" << this->m_Elastix->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel()
      << ".It" << this->m_Elastix->GetIterationCounter()
      << "." << resultImageFormat;

    /** Write the deformationFieldImage. */
    makeFileName1 << begin.str() << "deformationField" << end.str();
    typename DeformationFieldWriterType::Pointer deformationFieldWriter
      = DeformationFieldWriterType::New();
    deformationFieldWriter->SetFileName( makeFileName1.str().c_str() );
    deformationFieldWriter->SetInput( this->m_DeformationField );

    /** Do the writing. */
    try
    {
      deformationFieldWriter->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      /** Add information to the exception. */
      excp.SetLocation( "BSplineTransformWithDiffusion - DiffuseDeformationField()" );
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while writing the deformationField image.\n";
      excp.SetDescription( err_str );
      /** Print the exception. */
      xl::xout["error"] << excp << std::endl;
    }

    /** Write the GrayValueImage. */
    std::ostringstream makeFileName2( "" );
    makeFileName2 << begin.str() << "GrayValueImage" << end.str();
    typename GrayValueImageWriterType::Pointer grayValueImageWriter
      = GrayValueImageWriterType::New();
    grayValueImageWriter->SetFileName( makeFileName2.str().c_str() );
    if( this->m_AlsoFixed || this->m_UseFixedSegmentation )
    {
      grayValueImageWriter->SetInput( this->m_GrayValueImage2 );
    }
    else
    {
      grayValueImageWriter->SetInput( this->m_GrayValueImage1 );
    }

    /** Do the writing. */
    try
    {
      grayValueImageWriter->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      /** Add information to the exception. */
      excp.SetLocation( "BSplineTransformWithDiffusion - DiffuseDeformationField()" );
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while writing the grayValue image.\n";
      excp.SetDescription( err_str );
      /** Print the exception. */
      xl::xout["error"] << excp << std::endl;
    }

    /** Write the diffusedFieldImage. */
    std::ostringstream makeFileName3( "" );
    makeFileName3 << begin.str() << "diffusedField" << end.str();
    typename DeformationFieldWriterType::Pointer diffusedFieldWriter
      = DeformationFieldWriterType::New();
    diffusedFieldWriter->SetFileName( makeFileName3.str().c_str() );
    diffusedFieldWriter->SetInput( this->m_DiffusedField );

    /** Do the writing. */
    try
    {
      diffusedFieldWriter->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
      /** Add information to the exception. */
      excp.SetLocation( "BSplineTransformWithDiffusion - DiffuseDeformationField()" );
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while writing the diffusedField image.\n";
      excp.SetDescription( err_str );
      /** Print the exception. */
      xl::xout["error"] << excp << std::endl;
    }

  } // end if this->m_WriteDiffusionFiles

} // end DiffuseDeformationField()


/**
 * ******************* TransformPoint ******************
 */

template <class TElastix>
typename BSplineTransformWithDiffusion<TElastix>::OutputPointType
BSplineTransformWithDiffusion<TElastix>
::TransformPoint( const InputPointType  & point ) const
{
  return this->GenericDeformationFieldRegulizer::TransformPoint(point);

} // end TransformPoint()


} // end namespace elastix


#endif // end #ifndef __elxBSplineTransformWithDiffusion_HXX__

