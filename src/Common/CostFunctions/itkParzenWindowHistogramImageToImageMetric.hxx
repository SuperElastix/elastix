/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef _itkParzenWindowHistogramImageToImageMetric_HXX__
#define _itkParzenWindowHistogramImageToImageMetric_HXX__

#include "itkParzenWindowHistogramImageToImageMetric.h"

#include "itkBSplineDerivativeKernelFunction.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageScanlineIterator.h"
#include "vnl/vnl_math.h"

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template< class TFixedImage, class TMovingImage >
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ParzenWindowHistogramImageToImageMetric()
{
  this->m_NumberOfFixedHistogramBins        = 32;
  this->m_NumberOfMovingHistogramBins       = 32;
  this->m_JointPDF                          = 0;
  this->m_JointPDFDerivatives               = 0;
  this->m_FixedImageNormalizedMin           = 0.0;
  this->m_MovingImageNormalizedMin          = 0.0;
  this->m_FixedImageBinSize                 = 0.0;
  this->m_MovingImageBinSize                = 0.0;
  this->m_Alpha                             = 0.0;
  this->m_FixedIncrementalMarginalPDFRight  = 0;
  this->m_MovingIncrementalMarginalPDFRight = 0;
  this->m_FixedIncrementalMarginalPDFLeft   = 0;
  this->m_MovingIncrementalMarginalPDFLeft  = 0;

  this->m_FixedKernel                   = 0;
  this->m_MovingKernel                  = 0;
  this->m_DerivativeMovingKernel        = 0;
  this->m_FixedKernelBSplineOrder       = 0;
  this->m_MovingKernelBSplineOrder      = 3;
  this->m_FixedParzenTermToIndexOffset  = 0.5;
  this->m_MovingParzenTermToIndexOffset = -1.0;

  this->m_UseDerivative                 = false;
  this->m_UseFiniteDifferenceDerivative = false;
  this->m_FiniteDifferencePerturbation  = 1.0;

  this->SetUseImageSampler( true );
  this->SetUseFixedImageLimiter( true );
  this->SetUseMovingImageLimiter( true );

  this->m_UseExplicitPDFDerivatives = true;

  /** Initialize the m_ParzenWindowHistogramThreaderParameters */
  this->m_ParzenWindowHistogramThreaderParameters.m_Metric = this;

  // Multi-threading structs
  this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables     = NULL;
  this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariablesSize = 0;

} // end Constructor


/**
 * ******************* Destructor *******************
 */

template< class TFixedImage, class TMovingImage >
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::~ParzenWindowHistogramImageToImageMetric()
{
  delete[] this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables;
} // end Destructor


/**
 * ********************* PrintSelf ******************************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  /** Call the superclass' PrintSelf. */
  Superclass::PrintSelf( os, indent );

  /** Add debugging information. */
  os << indent << "NumberOfFixedHistogramBins: "
     << this->m_NumberOfFixedHistogramBins << std::endl;
  os << indent << "NumberOfMovingHistogramBins: "
     << this->m_NumberOfMovingHistogramBins << std::endl;
  os << indent << "FixedKernelBSplineOrder: "
     << this->m_FixedKernelBSplineOrder << std::endl;
  os << indent << "MovingKernelBSplineOrder: "
     << this->m_MovingKernelBSplineOrder << std::endl;

  /*double m_MovingImageNormalizedMin;
  double m_FixedImageNormalizedMin;
  double m_FixedImageBinSize;
  double m_MovingImageBinSize;
  double m_FixedParzenTermToIndexOffset;
  double m_MovingParzenTermToIndexOffset;
  bool m_UseDerivative;
  m_UseExplicitPDFDerivatives
  bool m_UseFiniteDifferenceDerivative;
  double m_FiniteDifferencePerturbation;*/

  /** This function is not complete, but we don't use it anyway. */

} // end PrintSelf()


/**
 * ********************* Initialize *****************************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::Initialize( void ) throw ( ExceptionObject )
{
  /** Call the superclass to check that standard components are available. */
  this->Superclass::Initialize();

  /** Set up the histograms. */
  this->InitializeHistograms();

  /** Set up the Parzen windows. */
  this->InitializeKernels();

  /** If the user plans to use a finite difference derivative,
   * allocate some memory for the perturbed alpha variables.
   */
  if( this->GetUseDerivative() && this->GetUseFiniteDifferenceDerivative() )
  {
    this->m_PerturbedAlphaRight.SetSize( this->GetNumberOfParameters() );
    this->m_PerturbedAlphaLeft.SetSize( this->GetNumberOfParameters() );
  }
  else
  {
    this->m_PerturbedAlphaRight.SetSize( 0 );
    this->m_PerturbedAlphaLeft.SetSize( 0 );
  }

} // end Initialize()


/**
 * ****************** InitializeHistograms *****************************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::InitializeHistograms( void )
{
  /* Compute binsize for the histogram.
   *
   * The binsize for the image intensities needs to be adjusted so that
   * we can avoid dealing with boundary conditions using the cubic
   * spline as the Parzen window.  We do this by increasing the size
   * of the bins so that the joint histogram becomes "padded" at the
   * borders. Because we are changing the binsize,
   * we also need to shift the minimum by the padded amount in order to
   * avoid minimum values filling in our padded region.
   *
   * Note that there can still be non-zero bin values in the padded region,
   * it's just that these bins will never be a central bin for the Parzen
   * window.
   */
  //int fixedPadding = 2;  // this will pad by 2 bins
  //int movingPadding = 2;  // this will pad by 2 bins
  int fixedPadding  = this->m_FixedKernelBSplineOrder / 2; // should be enough
  int movingPadding = this->m_MovingKernelBSplineOrder / 2;

  /** The ratio times the expected bin size will be added twice to the image range. */
  const double smallNumberRatio = 0.001;
  const double smallNumberFixed = smallNumberRatio
    * ( this->m_FixedImageMaxLimit - this->m_FixedImageMinLimit )
    / static_cast< double >( this->m_NumberOfFixedHistogramBins - 2 * fixedPadding - 1 );
  const double smallNumberMoving = smallNumberRatio
    * ( this->m_MovingImageMaxLimit - this->m_MovingImageMinLimit )
    / static_cast< double >( this->m_NumberOfFixedHistogramBins - 2 * movingPadding - 1 );

  /** Compute binsizes. */
  const double fixedHistogramWidth = static_cast< double >(
    static_cast< OffsetValueType >( this->m_NumberOfFixedHistogramBins ) // requires cast to signed type!
    - 2.0 * fixedPadding - 1.0 );
  this->m_FixedImageBinSize
    = ( this->m_FixedImageMaxLimit - this->m_FixedImageMinLimit
    + 2.0 * smallNumberFixed ) / fixedHistogramWidth;
  this->m_FixedImageBinSize = vnl_math_max( this->m_FixedImageBinSize, 1e-10 );
  this->m_FixedImageBinSize = vnl_math_min( this->m_FixedImageBinSize, 1e+10 );
  this->m_FixedImageNormalizedMin
    = ( this->m_FixedImageMinLimit - smallNumberFixed )
    / this->m_FixedImageBinSize - static_cast< double >( fixedPadding );

  const double movingHistogramWidth = static_cast< double >(
    static_cast< OffsetValueType >( this->m_NumberOfMovingHistogramBins ) // requires cast to signed type!
    - 2.0 * movingPadding - 1.0 );
  this->m_MovingImageBinSize
    = ( this->m_MovingImageMaxLimit - this->m_MovingImageMinLimit
    + 2.0 * smallNumberMoving ) / movingHistogramWidth;
  this->m_MovingImageBinSize = vnl_math_max( this->m_MovingImageBinSize, 1e-10 );
  this->m_MovingImageBinSize = vnl_math_min( this->m_MovingImageBinSize, 1e+10 );
  this->m_MovingImageNormalizedMin
    = ( this->m_MovingImageMinLimit - smallNumberMoving )
    / this->m_MovingImageBinSize - static_cast< double >( movingPadding );

  /** Allocate memory for the marginal PDF. */
  this->m_FixedImageMarginalPDF.SetSize( this->m_NumberOfFixedHistogramBins );
  this->m_MovingImageMarginalPDF.SetSize( this->m_NumberOfMovingHistogramBins );

  /** Allocate memory for the joint PDF and joint PDF derivatives. */

  /** First set these ones to zero */
  this->m_FixedIncrementalMarginalPDFRight  = 0;
  this->m_MovingIncrementalMarginalPDFRight = 0;
  this->m_FixedIncrementalMarginalPDFLeft   = 0;
  this->m_MovingIncrementalMarginalPDFLeft  = 0;

  /** For the joint PDF define a region starting from {0,0}
   * with size {this->m_NumberOfMovingHistogramBins, this->m_NumberOfFixedHistogramBins}
   * The dimension represents moving image Parzen window index
   * and fixed image Parzen window index, respectively.
   * The moving Parzen index is chosen as the first dimension,
   * because probably the moving B-spline kernel order will be larger
   * than the fixed B-spline kernel order and it is faster to iterate along
   * the first dimension.
   */
  this->m_JointPDF = JointPDFType::New();
  JointPDFRegionType jointPDFRegion;
  JointPDFIndexType  jointPDFIndex;
  JointPDFSizeType   jointPDFSize;
  jointPDFIndex.Fill( 0 );
  jointPDFSize[ 0 ] = this->m_NumberOfMovingHistogramBins;
  jointPDFSize[ 1 ] = this->m_NumberOfFixedHistogramBins;
  jointPDFRegion.SetIndex( jointPDFIndex );
  jointPDFRegion.SetSize( jointPDFSize );
  this->m_JointPDF->SetRegions( jointPDFRegion );
  this->m_JointPDF->Allocate();

  if( this->GetUseDerivative() )
  {
    /** For the derivatives of the joint PDF define a region starting from {0,0,0}
     * with size {GetNumberOfParameters(),m_NumberOfMovingHistogramBins,
     * m_NumberOfFixedHistogramBins}. The dimension represents transform parameters,
     * moving image Parzen window index and fixed image Parzen window index,
     * respectively.
     * For the incremental pdfs (used for finite difference derivative estimation)
     * the same size happens to be valid.
     */

    JointPDFDerivativesRegionType jointPDFDerivativesRegion;
    JointPDFDerivativesIndexType  jointPDFDerivativesIndex;
    JointPDFDerivativesSizeType   jointPDFDerivativesSize;
    jointPDFDerivativesIndex.Fill( 0 );
    jointPDFDerivativesSize[ 0 ] = this->GetNumberOfParameters();
    jointPDFDerivativesSize[ 1 ] = this->m_NumberOfMovingHistogramBins;
    jointPDFDerivativesSize[ 2 ] = this->m_NumberOfFixedHistogramBins;
    jointPDFDerivativesRegion.SetIndex( jointPDFDerivativesIndex );
    jointPDFDerivativesRegion.SetSize( jointPDFDerivativesSize );

    if( this->GetUseFiniteDifferenceDerivative() )
    {
      this->m_JointPDFDerivatives = 0;

      this->m_IncrementalJointPDFRight = JointPDFDerivativesType::New();
      this->m_IncrementalJointPDFLeft  = JointPDFDerivativesType::New();
      this->m_IncrementalJointPDFRight->SetRegions( jointPDFDerivativesRegion );
      this->m_IncrementalJointPDFLeft->SetRegions( jointPDFDerivativesRegion );
      this->m_IncrementalJointPDFRight->Allocate();
      this->m_IncrementalJointPDFLeft->Allocate();

      /** Also initialize the incremental marginal pdfs. */
      IncrementalMarginalPDFRegionType fixedIMPDFRegion;
      IncrementalMarginalPDFIndexType  fixedIMPDFIndex;
      IncrementalMarginalPDFSizeType   fixedIMPDFSize;

      IncrementalMarginalPDFRegionType movingIMPDFRegion;
      IncrementalMarginalPDFIndexType  movingIMPDFIndex;
      IncrementalMarginalPDFSizeType   movingIMPDFSize;

      fixedIMPDFIndex.Fill( 0 );
      fixedIMPDFSize[ 0 ] = this->GetNumberOfParameters();
      fixedIMPDFSize[ 1 ] = this->m_NumberOfFixedHistogramBins;
      fixedIMPDFRegion.SetSize( fixedIMPDFSize );
      fixedIMPDFRegion.SetIndex( fixedIMPDFIndex );

      movingIMPDFIndex.Fill( 0 );
      movingIMPDFSize[ 0 ] = this->GetNumberOfParameters();
      movingIMPDFSize[ 1 ] = this->m_NumberOfMovingHistogramBins;
      movingIMPDFRegion.SetSize( movingIMPDFSize );
      movingIMPDFRegion.SetIndex( movingIMPDFIndex );

      this->m_FixedIncrementalMarginalPDFRight  = IncrementalMarginalPDFType::New();
      this->m_MovingIncrementalMarginalPDFRight = IncrementalMarginalPDFType::New();
      this->m_FixedIncrementalMarginalPDFLeft   = IncrementalMarginalPDFType::New();
      this->m_MovingIncrementalMarginalPDFLeft  = IncrementalMarginalPDFType::New();

      this->m_FixedIncrementalMarginalPDFRight->SetRegions( fixedIMPDFRegion );
      this->m_MovingIncrementalMarginalPDFRight->SetRegions( movingIMPDFRegion );
      this->m_FixedIncrementalMarginalPDFLeft->SetRegions( fixedIMPDFRegion );
      this->m_MovingIncrementalMarginalPDFLeft->SetRegions( movingIMPDFRegion );

      this->m_FixedIncrementalMarginalPDFRight->Allocate();
      this->m_MovingIncrementalMarginalPDFRight->Allocate();
      this->m_FixedIncrementalMarginalPDFLeft->Allocate();
      this->m_MovingIncrementalMarginalPDFLeft->Allocate();
    } // end if this->GetUseFiniteDifferenceDerivative()
    else
    {
      if( this->m_UseExplicitPDFDerivatives )
      {
        this->m_IncrementalJointPDFRight = 0;
        this->m_IncrementalJointPDFLeft  = 0;

        this->m_JointPDFDerivatives = JointPDFDerivativesType::New();
        this->m_JointPDFDerivatives->SetRegions( jointPDFDerivativesRegion );
        this->m_JointPDFDerivatives->Allocate();
      }
      else
      {
        /** De-allocate large amount of memory for the m_JointPDFDerivatives. */
        // \todo Should not be allocated in the first place
        if( !this->m_JointPDFDerivatives.IsNull() )
        {
          jointPDFDerivativesSize.Fill( 0 );
          jointPDFDerivativesRegion.SetSize( jointPDFDerivativesSize );
          this->m_JointPDFDerivatives->SetRegions( jointPDFDerivativesRegion );
          this->m_JointPDFDerivatives->Allocate();
          this->m_JointPDFDerivatives->GetPixelContainer()->Squeeze();
        }
      }
    }
  }
  else
  {
    this->m_JointPDFDerivatives      = 0;
    this->m_IncrementalJointPDFRight = 0;
    this->m_IncrementalJointPDFLeft  = 0;
  }

} // end InitializeHistograms()


/**
 * ****************** InitializeKernels *****************************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::InitializeKernels( void )
{
  switch( this->m_FixedKernelBSplineOrder )
  {
    case 0:
      this->m_FixedKernel = BSplineKernelFunction< 0 >::New(); break;
    case 1:
      this->m_FixedKernel = BSplineKernelFunction< 1 >::New(); break;
    case 2:
      this->m_FixedKernel = BSplineKernelFunction< 2 >::New(); break;
    case 3:
      this->m_FixedKernel = BSplineKernelFunction< 3 >::New(); break;
    default:
      itkExceptionMacro( << "The following FixedKernelBSplineOrder is not implemented: " \
                         << this->m_FixedKernelBSplineOrder );
  } // end switch FixedKernelBSplineOrder

  switch( this->m_MovingKernelBSplineOrder )
  {
    case 0:
      this->m_MovingKernel = BSplineKernelFunction< 0 >::New();
      /** The derivative of a zero order B-spline makes no sense. Using the
       * derivative of a first order gives a kind of finite difference idea
       * Anyway, if you plan to call GetValueAndDerivative you should use
       * a higher B-spline order.
       */
      this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction< 1 >::New();
      break;
    case 1:
      this->m_MovingKernel           = BSplineKernelFunction< 1 >::New();
      this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction< 1 >::New();
      break;
    case 2:
      this->m_MovingKernel           = BSplineKernelFunction< 2 >::New();
      this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction< 2 >::New();
      break;
    case 3:
      this->m_MovingKernel           = BSplineKernelFunction< 3 >::New();
      this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction< 3 >::New();
      break;
    default:
      itkExceptionMacro( << "The following MovingKernelBSplineOrder is not implemented: " \
                         << this->m_MovingKernelBSplineOrder );
  } // end switch MovingKernelBSplineOrder

  /** The region of support of the Parzen window determines which bins
   * of the joint PDF are effected by the pair of image values.
   * For example, if we are using a cubic spline for the moving image Parzen
   * window, four bins are affected. If the fixed image Parzen window is
   * a zero-order spline (box car) only one bin is affected.
   */

  /** Set the size of the Parzen window. */
  JointPDFSizeType parzenWindowSize;
  parzenWindowSize[ 0 ] = this->m_MovingKernelBSplineOrder + 1;
  parzenWindowSize[ 1 ] = this->m_FixedKernelBSplineOrder + 1;
  this->m_JointPDFWindow.SetSize( parzenWindowSize );
  this->m_JointPDFWindow.SetSize( parzenWindowSize );

  /** The ParzenIndex is the lowest bin number that is affected by a
   * pixel and computed as:
   * ParzenIndex = vcl_floor( ParzenTerm + ParzenTermToIndexOffset )
   * where ParzenTermToIndexOffset = 1/2, 0, -1/2, or -1.
   */
  this->m_FixedParzenTermToIndexOffset
    = 0.5 - static_cast< double >( this->m_FixedKernelBSplineOrder ) / 2.0;
  this->m_MovingParzenTermToIndexOffset
    = 0.5 - static_cast< double >( this->m_MovingKernelBSplineOrder ) / 2.0;

} // end InitializeKernels()


/**
 * ********************* InitializeThreadingParameters ****************************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::InitializeThreadingParameters( void ) const
{
  /** Call superclass implementation. */
  Superclass::InitializeThreadingParameters();

  /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   * Filling the potentially large vectors is performed later, in each thread,
   * which has performance benefits for larger vector sizes.
   */

  /** Construct regions for the joint histograms. */
  JointPDFRegionType jointPDFRegion;
  JointPDFIndexType  jointPDFIndex;
  JointPDFSizeType   jointPDFSize;
  jointPDFIndex.Fill( 0 );
  jointPDFSize[ 0 ] = this->m_NumberOfMovingHistogramBins;
  jointPDFSize[ 1 ] = this->m_NumberOfFixedHistogramBins;
  jointPDFRegion.SetIndex( jointPDFIndex );
  jointPDFRegion.SetSize( jointPDFSize );

  /** Only resize the array of structs when needed. */
  if( this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariablesSize != this->m_NumberOfThreads )
  {
    delete[] this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables;
    this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables
      = new AlignedParzenWindowHistogramGetValueAndDerivativePerThreadStruct[
      this->m_NumberOfThreads ];
    this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariablesSize = this->m_NumberOfThreads;
  }

  /** Some initialization. */
  for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
  {
    this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted = NumericTraits< SizeValueType >::Zero;

    // Initialize the joint pdf
    JointPDFPointer & jointPDF = this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables[ i ].st_JointPDF;
    if( jointPDF.IsNull() ) { jointPDF = JointPDFType::New(); }
    if( jointPDF->GetLargestPossibleRegion() != jointPDFRegion )
    {
      jointPDF->SetRegions( jointPDFRegion );
      jointPDF->Allocate();
    }
  }

} // end InitializeThreadingParameters()


/**
 * ******************** GetDerivative ***************************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::GetDerivative( const ParametersType & parameters, DerivativeType & derivative ) const
{
  /** Call the combined version, since the additional computation of
   * the value does not take extra time.
   */
  MeasureType value;
  this->GetValueAndDerivative( parameters, value, derivative );

} // end GetDerivative()


/**
 * ******************** GetValueAndDerivative ***************************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivative( const ParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  if( this->GetUseFiniteDifferenceDerivative() )
  {
    this->GetValueAndFiniteDifferenceDerivative( parameters, value, derivative );
  }
  else
  {
    this->GetValueAndAnalyticDerivative( parameters, value, derivative );
  }
} // end GetValueAndDerivative()


/**
 * ********************** EvaluateParzenValues ***************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::EvaluateParzenValues(
  double parzenWindowTerm, OffsetValueType parzenWindowIndex,
  const KernelFunctionType * kernel, ParzenValueContainerType & parzenValues ) const
{
  const unsigned int max_i = parzenValues.GetSize();
  for( unsigned int i = 0; i < max_i; ++i, ++parzenWindowIndex )
  {
    parzenValues[ i ] = kernel->Evaluate(
      static_cast< double >( parzenWindowIndex ) - parzenWindowTerm );
  }
} // end EvaluateParzenValues()


/**
 * ********************** UpdateJointPDFAndDerivatives ***************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::UpdateJointPDFAndDerivatives(
  const RealType & fixedImageValue,
  const RealType & movingImageValue,
  const DerivativeType * imageJacobian,
  const NonZeroJacobianIndicesType * nzji,
  JointPDFType * jointPDF ) const
{
  typedef ImageScanlineIterator< JointPDFType > PDFIteratorType;

  /** Determine Parzen window arguments (see eq. 6 of Mattes paper [2]). */
  const double fixedImageParzenWindowTerm
    = fixedImageValue / this->m_FixedImageBinSize - this->m_FixedImageNormalizedMin;
  const double movingImageParzenWindowTerm
    = movingImageValue / this->m_MovingImageBinSize - this->m_MovingImageNormalizedMin;

  /** The lowest bin numbers affected by this pixel: */
  const OffsetValueType fixedImageParzenWindowIndex
    = static_cast< OffsetValueType >( vcl_floor(
    fixedImageParzenWindowTerm + this->m_FixedParzenTermToIndexOffset ) );
  const OffsetValueType movingImageParzenWindowIndex
    = static_cast< OffsetValueType >( vcl_floor(
    movingImageParzenWindowTerm + this->m_MovingParzenTermToIndexOffset ) );

  /** The Parzen values. */
  ParzenValueContainerType fixedParzenValues( this->m_JointPDFWindow.GetSize()[ 1 ] );
  ParzenValueContainerType movingParzenValues( this->m_JointPDFWindow.GetSize()[ 0 ] );
  this->EvaluateParzenValues(
    fixedImageParzenWindowTerm, fixedImageParzenWindowIndex,
    this->m_FixedKernel, fixedParzenValues );
  this->EvaluateParzenValues(
    movingImageParzenWindowTerm, movingImageParzenWindowIndex,
    this->m_MovingKernel, movingParzenValues );

  /** Position the JointPDFWindow. */
  JointPDFIndexType pdfWindowIndex;
  pdfWindowIndex[ 0 ] = movingImageParzenWindowIndex;
  pdfWindowIndex[ 1 ] = fixedImageParzenWindowIndex;
  /** For thread-safety, make a local copy of the support region,
   * and use that one. Because each thread will modify it.
   */
  JointPDFRegionType jointPDFWindow = this->m_JointPDFWindow;
  jointPDFWindow.SetIndex( pdfWindowIndex );
  PDFIteratorType it( jointPDF, jointPDFWindow );

  if( !imageJacobian )
  {
    /** Loop over the Parzen window region and increment the values. */
    for( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
    {
      const double fv = fixedParzenValues[ f ];
      for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
      {
        it.Value() += static_cast< PDFValueType >( fv * movingParzenValues[ m ] );
        ++it;
      }
      it.NextLine();
    }
  }
  else
  {
    /** Compute the derivatives of the moving Parzen window. */
    ParzenValueContainerType derivativeMovingParzenValues(
    this->m_JointPDFWindow.GetSize()[ 0 ] );
    this->EvaluateParzenValues(
      movingImageParzenWindowTerm, movingImageParzenWindowIndex,
      this->m_DerivativeMovingKernel, derivativeMovingParzenValues );

    const double et = static_cast< double >( this->m_MovingImageBinSize );

    /** Loop over the Parzen window region and increment the values
    * Also update the pdf derivatives.
    */
    for( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
    {
      const double fv    = fixedParzenValues[ f ];
      const double fv_et = fv / et;
      for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
      {
        it.Value() += static_cast< PDFValueType >( fv * movingParzenValues[ m ] );
        this->UpdateJointPDFDerivatives(
          it.GetIndex(), fv_et * derivativeMovingParzenValues[ m ],
          *imageJacobian, *nzji );
        ++it;
      }
      it.NextLine();
    }
  }

} // end UpdateJointPDFAndDerivatives()


/**
 * *************** UpdateJointPDFDerivatives ***************************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::UpdateJointPDFDerivatives(
  const JointPDFIndexType & pdfIndex, double factor,
  const DerivativeType & imageJacobian,
  const NonZeroJacobianIndicesType & nzji ) const
{
  /** Get the pointer to the element with index [0, pdfIndex[0], pdfIndex[1]]. */
  PDFDerivativeValueType * derivPtr = this->m_JointPDFDerivatives->GetBufferPointer()
    + ( pdfIndex[ 0 ] * this->m_JointPDFDerivatives->GetOffsetTable()[ 1 ] )
    + ( pdfIndex[ 1 ] * this->m_JointPDFDerivatives->GetOffsetTable()[ 2 ] );

  if( nzji.size() == this->GetNumberOfParameters() )
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjac = imageJacobian.begin();
    for( unsigned int mu = 0; mu < this->GetNumberOfParameters(); ++mu )
    {
      *( derivPtr ) -= static_cast< PDFDerivativeValueType >( ( *imjac ) * factor );
      ++derivPtr;
      ++imjac;
    }
  }
  else
  {
    /** Loop only over the non-zero Jacobians. */
    for( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
    {
      const unsigned int       mu  = nzji[ i ];
      PDFDerivativeValueType * ptr = derivPtr + mu;
      *( ptr ) -= static_cast< PDFDerivativeValueType >( imageJacobian[ i ] * factor );
    }
  }

} // end UpdateJointPDFDerivatives()


/**
 * *********************** NormalizeJointPDF ***********************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::NormalizeJointPDF( JointPDFType * pdf, const double & factor ) const
{
  typedef ImageScanlineIterator< JointPDFType > JointPDFIteratorType;
  JointPDFIteratorType it( pdf, pdf->GetBufferedRegion() );
  const PDFValueType   castfac = static_cast< PDFValueType >( factor );
  while( !it.IsAtEnd() )
  {
    while( !it.IsAtEndOfLine() )
    {
      it.Value() *= castfac;
      ++it;
    }
    it.NextLine();
  }

} // end NormalizeJointPDF()


/**
 * *********************** NormalizeJointPDFDerivatives ***********************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::NormalizeJointPDFDerivatives( JointPDFDerivativesType * pdf, const double & factor ) const
{
  typedef ImageScanlineIterator< JointPDFDerivativesType > JointPDFDerivativesIteratorType;
  JointPDFDerivativesIteratorType it( pdf, pdf->GetBufferedRegion() );
  const PDFValueType              castfac = static_cast< PDFValueType >( factor );
  while( !it.IsAtEnd() )
  {
    while( !it.IsAtEndOfLine() )
    {
      it.Value() *= castfac;
      ++it;
    }
    it.NextLine();
  }

} // end NormalizeJointPDFDerivatives()


/**
 * ************************ ComputeMarginalPDF ***********************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ComputeMarginalPDF(
  const JointPDFType * itkNotUsed( jointPDF ),
  MarginalPDFType & marginalPDF, const unsigned int & direction ) const
{
  typedef ImageLinearIteratorWithIndex< JointPDFType > JointPDFLinearIterator;
  // \todo: bug? shouldn't this be over the function argument jointPDF ?
  JointPDFLinearIterator linearIter( this->m_JointPDF, this->m_JointPDF->GetBufferedRegion() );
  linearIter.SetDirection( direction );
  linearIter.GoToBegin();
  unsigned int marginalIndex = 0;
  while( !linearIter.IsAtEnd() )
  {
    PDFValueType sum = 0.0;
    while( !linearIter.IsAtEndOfLine() )
    {
      sum += linearIter.Get();
      ++linearIter;
    }
    marginalPDF[ marginalIndex ] = sum;
    linearIter.NextLine();
    ++marginalIndex;
  }

} // end ComputeMarginalPDFs()


/**
 * ******************** ComputeIncrementalMarginalPDFs *******************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ComputeIncrementalMarginalPDFs(
  const JointPDFDerivativesType * incrementalPDF,
  IncrementalMarginalPDFType * fixedIncrementalMarginalPDF,
  IncrementalMarginalPDFType * movingIncrementalMarginalPDF ) const
{
  typedef itk::ImageRegionConstIterator< JointPDFDerivativesType >        IncIteratorType;
  typedef itk::ImageLinearIteratorWithIndex< IncrementalMarginalPDFType > IncMargIteratorType;

  fixedIncrementalMarginalPDF->FillBuffer( itk::NumericTraits< PDFValueType >::Zero );
  movingIncrementalMarginalPDF->FillBuffer( itk::NumericTraits< PDFValueType >::Zero );

  IncIteratorType     incit( incrementalPDF, incrementalPDF->GetLargestPossibleRegion() );
  IncMargIteratorType fixincit( fixedIncrementalMarginalPDF,
  fixedIncrementalMarginalPDF->GetLargestPossibleRegion() );
  IncMargIteratorType movincit( movingIncrementalMarginalPDF,
  movingIncrementalMarginalPDF->GetLargestPossibleRegion() );

  incit.GoToBegin();
  fixincit.GoToBegin();
  movincit.GoToBegin();

  /** Loop over the incremental pdf and update the incremental marginal pdfs. */
  for( unsigned int f = 0; f < this->m_NumberOfFixedHistogramBins; ++f )
  {
    for( unsigned int m = 0; m < this->m_NumberOfMovingHistogramBins; ++m )
    {
      for( unsigned int p = 0; p < this->GetNumberOfParameters(); ++p )
      {
        fixincit.Value() += incit.Get();
        movincit.Value() += incit.Get();
        ++incit;
        ++fixincit;
        ++movincit;
      }
      fixincit.GoToBeginOfLine();
      movincit.NextLine();
    }
    fixincit.NextLine();
    movincit.GoToBegin();
  }

} // end ComputeIncrementalMarginalPDFs()


/**
 * ******************* UpdateJointPDFAndIncrementalPDFs *******************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::UpdateJointPDFAndIncrementalPDFs(
  RealType fixedImageValue, RealType movingImageValue, RealType movingMaskValue,
  const DerivativeType & movingImageValuesRight,
  const DerivativeType & movingImageValuesLeft,
  const DerivativeType & movingMaskValuesRight,
  const DerivativeType & movingMaskValuesLeft,
  const NonZeroJacobianIndicesType & nzji ) const
{
  /** Pointers to the first pixels in the incremental joint pdfs. */
  PDFDerivativeValueType * incRightBasePtr = this->m_IncrementalJointPDFRight->GetBufferPointer();
  PDFDerivativeValueType * incLeftBasePtr  = this->m_IncrementalJointPDFLeft->GetBufferPointer();

  /** The Parzen value containers. */
  ParzenValueContainerType fixedParzenValues( this->m_JointPDFWindow.GetSize()[ 1 ] );
  ParzenValueContainerType movingParzenValues( this->m_JointPDFWindow.GetSize()[ 0 ] );

  /** Determine fixed image Parzen window arguments (see eq. 6 of Mattes paper [2]). */
  const double fixedImageParzenWindowTerm
    = fixedImageValue / this->m_FixedImageBinSize - this->m_FixedImageNormalizedMin;

  /** The lowest bin numbers affected by this pixel: */
  const OffsetValueType fixedImageParzenWindowIndex
    = static_cast< OffsetValueType >( vcl_floor(
    fixedImageParzenWindowTerm + this->m_FixedParzenTermToIndexOffset ) );
  this->EvaluateParzenValues(
    fixedImageParzenWindowTerm, fixedImageParzenWindowIndex,
    this->m_FixedKernel, fixedParzenValues );

  if( movingMaskValue > 1e-10 )
  {
    /** Determine moving image Parzen window arguments (see eq. 6 of Mattes paper [2]). */
    const double movingImageParzenWindowTerm
      = movingImageValue / this->m_MovingImageBinSize - this->m_MovingImageNormalizedMin;
    const OffsetValueType movingImageParzenWindowIndex
      = static_cast< OffsetValueType >( vcl_floor(
      movingImageParzenWindowTerm + this->m_MovingParzenTermToIndexOffset ) );
    this->EvaluateParzenValues(
      movingImageParzenWindowTerm, movingImageParzenWindowIndex,
      this->m_MovingKernel, movingParzenValues );

    /** Position the JointPDFWindow (set the start index). */
    JointPDFIndexType pdfIndex;
    pdfIndex[ 0 ] = movingImageParzenWindowIndex;
    pdfIndex[ 1 ] = fixedImageParzenWindowIndex;

    /** Loop over the Parzen window region and do the following update:
     *
     * m_JointPDF(M,F) += movingMask * fixedParzen(F) * movingParzen(M);
     * m_IncrementalJointPDF<Right/Left>(k,M,F) -= movingMask * fixedParzen(F) * movingParzen(M);
     * for all k with nonzero Jacobian.
     */
    for( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
    {
      const double fv_mask = fixedParzenValues[ f ] * movingMaskValue;
      for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
      {
        const PDFValueType fv_mask_mv
                                                = static_cast< PDFValueType >( fv_mask * movingParzenValues[ m ] );
        this->m_JointPDF->GetPixel( pdfIndex ) += fv_mask_mv;

        unsigned long offset = static_cast< unsigned long >(
          pdfIndex[ 0 ] * this->m_IncrementalJointPDFRight->GetOffsetTable()[ 1 ]
          + pdfIndex[ 1 ] * this->m_IncrementalJointPDFRight->GetOffsetTable()[ 2 ] );

        /** Get the pointer to the element with index [0, pdfIndex[0], pdfIndex[1]]. */
        PDFDerivativeValueType * incRightPtr = incRightBasePtr + offset;
        PDFDerivativeValueType * incLeftPtr  = incLeftBasePtr + offset;

        /** Loop only over the non-zero Jacobians. */
        for( unsigned int i = 0; i < nzji.size(); ++i )
        {
          const unsigned int       mu   = nzji[ i ];
          PDFDerivativeValueType * rPtr = incRightPtr + mu;
          PDFDerivativeValueType * lPtr = incLeftPtr + mu;
          *( rPtr ) -= fv_mask_mv;
          *( lPtr ) -= fv_mask_mv;
        } // end for i

        ++( pdfIndex[ 0 ] );
      } // end for m

      pdfIndex[ 0 ] = movingImageParzenWindowIndex;
      ++( pdfIndex[ 1 ] );

    } // end for f

  } // end if movingMaskValue > 1e-10

  /** Loop only over the non-zero Jacobians and update the incremental pdfs and
   * update the perturbed alphas:
   *
   * m_IncrementalJointPDF<Right/Left>(k,M,F) +=
   *   movingMask<Right/Left>[k] * fixedParzen(F) * movingParzen<Right/Left>(M)[k];
   * m_PerturbedAlpha<Right/Left>[k] += movingMask<Right/Left>[k] - movingMask;
   * for all k with nonzero Jacobian.
   */
  JointPDFDerivativesIndexType rindex;
  JointPDFDerivativesIndexType lindex;
  for( unsigned int i = 0; i < nzji.size(); ++i )
  {
    const unsigned int mu    = nzji[ i ];
    const double       maskr = movingMaskValuesRight[ i ];
    const double       maskl = movingMaskValuesLeft[ i ];

    if( maskr > 1e-10 )
    {
      /** Compute Parzen stuff; note: we reuse the movingParzenValues container. */
      const double movr = movingImageValuesRight[ i ];
      const double movParzenWindowTermRight
        = movr / this->m_MovingImageBinSize - this->m_MovingImageNormalizedMin;
      const OffsetValueType movParzenWindowIndexRight
        = static_cast< OffsetValueType >( vcl_floor(
        movParzenWindowTermRight + this->m_MovingParzenTermToIndexOffset ) );
      this->EvaluateParzenValues(
        movParzenWindowTermRight, movParzenWindowIndexRight,
        this->m_MovingKernel, movingParzenValues );

      /** Initialize index in IncrementalJointPDFRight. */
      rindex[ 0 ] = mu;
      rindex[ 1 ] = movParzenWindowIndexRight;
      rindex[ 2 ] = fixedImageParzenWindowIndex;

      /** Loop over Parzen window and update IncrementalJointPDFRight. */
      for( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
      {
        const double fv_mask = fixedParzenValues[ f ] * maskr;
        for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
        {
          const PDFValueType fv_mask_mv = static_cast< PDFValueType >( fv_mask * movingParzenValues[ m ] );
          this->m_IncrementalJointPDFRight->GetPixel( rindex ) += fv_mask_mv;
          ++( rindex[ 1 ] );
        } // end for m

        ++( rindex[ 2 ] );
        rindex[ 1 ] = movParzenWindowIndexRight;

      } // end for f
    }   // end if maskr

    if( maskl > 1e-10 )
    {
      /** Compute Parzen stuff; note: we reuse the movingParzenValues container. */
      const double movl = movingImageValuesLeft[ i ];
      const double movParzenWindowTermLeft
        = movl / this->m_MovingImageBinSize - this->m_MovingImageNormalizedMin;
      const OffsetValueType movParzenWindowIndexLeft
        = static_cast< OffsetValueType >( vcl_floor(
        movParzenWindowTermLeft + this->m_MovingParzenTermToIndexOffset ) );
      this->EvaluateParzenValues(
        movParzenWindowTermLeft, movParzenWindowIndexLeft,
        this->m_MovingKernel, movingParzenValues );

      /** Initialize index in IncrementalJointPDFLeft. */
      lindex[ 0 ] = mu;
      lindex[ 1 ] = movParzenWindowIndexLeft;
      lindex[ 2 ] = fixedImageParzenWindowIndex;

      /** Loop over Parzen window and update IncrementalJointPDFLeft. */
      for( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
      {
        const double fv_mask = fixedParzenValues[ f ] * maskl;
        for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
        {
          const PDFValueType fv_mask_mv = static_cast< PDFValueType >( fv_mask * movingParzenValues[ m ] );
          this->m_IncrementalJointPDFLeft->GetPixel( lindex ) += fv_mask_mv;
          ++( lindex[ 1 ] );
        } // end for m

        ++( lindex[ 2 ] );
        lindex[ 1 ] = movParzenWindowIndexLeft;

      } // end for f
    }   // end if maskl

    /** Update the perturbed alphas. */
    this->m_PerturbedAlphaRight[ mu ] += ( maskr - movingMaskValue );
    this->m_PerturbedAlphaLeft[ mu ]  += ( maskl - movingMaskValue );
  } // end for i

} // end UpdateJointPDFAndIncrementalPDFs()


/**
 * ************************ ComputePDFsSingleThreaded **************************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ComputePDFsSingleThreaded( const ParametersType & parameters ) const
{
  /** Initialize some variables. */
  this->m_JointPDF->FillBuffer( 0.0 );
  this->m_NumberOfPixelsCounted = 0;
  this->m_Alpha                 = 0.0;

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();

  /** Loop over sample container and compute contribution of each sample to pdfs. */
  for( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImagePointType        mappedPoint;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value and check if the point is
    * inside the moving image buffer.
    */
    if( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, 0 );
    }

    if( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      RealType fixedImageValue = static_cast< RealType >( ( *fiter ).Value().m_ImageValue );

      /** Make sure the values fall within the histogram range. */
      fixedImageValue  = this->GetFixedImageLimiter()->Evaluate( fixedImageValue );
      movingImageValue = this->GetMovingImageLimiter()->Evaluate( movingImageValue );

      /** Compute this sample's contribution to the joint distributions. */
      this->UpdateJointPDFAndDerivatives(
        fixedImageValue, movingImageValue, 0, 0, this->m_JointPDF.GetPointer() );
    }

  } // end iterating over fixed image spatial sample container for loop

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Compute alpha. */
  this->m_Alpha = 1.0 / static_cast< double >( this->m_NumberOfPixelsCounted );

} // end ComputePDFsSingleThreaded()


/**
 * ************************ ComputePDFs **************************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ComputePDFs( const ParametersType & parameters ) const
{
  /** Option for now to still use the single threaded code. */
  if( !this->m_UseMultiThread )
  {
    return this->ComputePDFsSingleThreaded( parameters );
  }

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Launch multi-threading JointPDF computation. */
  this->LaunchComputePDFsThreaderCallback();

  /** Gather the results from all threads. */
  this->AfterThreadedComputePDFs();

} // end ComputePDFs()


/**
 * ******************* ThreadedComputePDFs *******************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ThreadedComputePDFs( ThreadIdType threadId )
{
  /** Get a handle to the pre-allocated joint PDF for the current thread.
   * The initialization is performed here, so that it is done multi-threadedly
   * instead of sequentially in InitializeThreadingParameters().
   */
  JointPDFPointer & jointPDF = this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables[ threadId ].st_JointPDF;
  jointPDF->FillBuffer( NumericTraits< PDFValueType >::Zero );

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer     = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads
    = static_cast< unsigned long >( vcl_ceil( static_cast< double >( sampleContainerSize )
    / static_cast< double >( this->m_NumberOfThreads ) ) );

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end   = nrOfSamplesPerThreads * ( threadId + 1 );
  pos_begin = ( pos_begin > sampleContainerSize ) ? sampleContainerSize : pos_begin;
  pos_end   = ( pos_end > sampleContainerSize ) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->Begin();
  fbegin                                                 += (int)pos_begin;
  fend                                                   += (int)pos_end;

  /** Create variables to store intermediate results. circumvent false sharing */
  unsigned long numberOfPixelsCounted = 0;

  /** Loop over sample container and compute contribution of each sample to pdfs. */
  for( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImagePointType        mappedPoint;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value and check if the point is
     * inside the moving image buffer.
     */
    if( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, 0 );
    }

    if( sampleOk )
    {
      numberOfPixelsCounted++;

      /** Get the fixed image value. */
      RealType fixedImageValue = static_cast< RealType >( ( *fiter ).Value().m_ImageValue );

      /** Make sure the values fall within the histogram range. */
      fixedImageValue  = this->GetFixedImageLimiter()->Evaluate( fixedImageValue );
      movingImageValue = this->GetMovingImageLimiter()->Evaluate( movingImageValue );

      /** Compute this sample's contribution to the joint distributions. */
      this->UpdateJointPDFAndDerivatives(
        fixedImageValue, movingImageValue, 0, 0,
        jointPDF.GetPointer() );
    }
  } // end iterating over fixed image spatial sample container for loop

  /** Only update these variables at the end to prevent unnecessary "false sharing". */
  this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCounted = numberOfPixelsCounted;

} // end ThreadedComputePDFs()


/**
 * ******************* AfterThreadedComputePDFs *******************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::AfterThreadedComputePDFs( void ) const
{
  /** Accumulate the number of pixels. */
  this->m_NumberOfPixelsCounted
    = this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables[ 0 ].st_NumberOfPixelsCounted;
  for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
  {
    this->m_NumberOfPixelsCounted
      += this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted;

    /** Reset this variable for the next iteration. */
    this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted = 0;
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Compute alpha. */
  this->m_Alpha = 1.0 / static_cast< double >( this->m_NumberOfPixelsCounted );

  /** Accumulate joint histogram. */
  // could be multi-threaded too, by each thread updating only a part of the JointPDF.
  typedef ImageScanlineIterator< JointPDFType > JointPDFIteratorType;
  JointPDFIteratorType                it( this->m_JointPDF, this->m_JointPDF->GetBufferedRegion() );
  std::vector< JointPDFIteratorType > itT( this->m_NumberOfThreads );
  for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
  {
    itT[ i ] = JointPDFIteratorType(
      this->m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables[ i ].st_JointPDF,
      this->m_JointPDF->GetBufferedRegion() );
  }

  PDFValueType sum;
  while( !it.IsAtEnd() )
  {
    while( !it.IsAtEndOfLine() )
    {
      sum = NumericTraits< PDFValueType >::Zero;
      for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
      {
        sum += itT[ i ].Value();
        ++itT[ i ];
      }
      it.Set( sum );
      ++it;
    }
    it.NextLine();
    for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
    {
      itT[ i ].NextLine();
    }
  }

} // end AfterThreadedComputePDFs()


/**
 * **************** ComputePDFsThreaderCallback *******
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ComputePDFsThreaderCallback( void * arg )
{
  ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
  ThreadIdType     threadId   = infoStruct->ThreadID;

  ParzenWindowHistogramMultiThreaderParameterType * temp
    = static_cast< ParzenWindowHistogramMultiThreaderParameterType * >( infoStruct->UserData );

  temp->m_Metric->ThreadedComputePDFs( threadId );

  return ITK_THREAD_RETURN_VALUE;

} // end ComputePDFsThreaderCallback()


/**
 * *********************** LaunchComputePDFsThreaderCallback***************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::LaunchComputePDFsThreaderCallback( void ) const
{
  /** Setup local threader. */
  // \todo: is a global threader better performance-wise? check
  typename ThreaderType::Pointer local_threader = ThreaderType::New();
  local_threader->SetNumberOfThreads( this->m_NumberOfThreads );
  local_threader->SetSingleMethod( this->ComputePDFsThreaderCallback,
    const_cast< void * >( static_cast< const void * >(
      &this->m_ParzenWindowHistogramThreaderParameters ) ) );

  /** Launch. */
  local_threader->SingleMethodExecute();

} // end LaunchComputePDFsThreaderCallback()


/**
 * ************************ ComputePDFsAndPDFDerivatives *******************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ComputePDFsAndPDFDerivatives( const ParametersType & parameters ) const
{
  /** Initialize some variables. */
  this->m_JointPDF->FillBuffer( 0.0 );
  this->m_JointPDFDerivatives->FillBuffer( 0.0 );
  this->m_Alpha                 = 0.0;
  this->m_NumberOfPixelsCounted = 0;

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType             imageJacobian( nzji.size() );
  TransformJacobianType      jacobian;

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();

  /** Loop over sample container and compute contribution of each sample to pdfs. */
  for( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImagePointType        mappedPoint;
    MovingImageDerivativeType   movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    if( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative );
    }

    if( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      RealType fixedImageValue = static_cast< RealType >( ( *fiter ).Value().m_ImageValue );

      /** Make sure the values fall within the histogram range. */
      fixedImageValue  = this->GetFixedImageLimiter()->Evaluate( fixedImageValue );
      movingImageValue = this->GetMovingImageLimiter()->Evaluate(
        movingImageValue, movingImageDerivative );

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner product (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Update the joint pdf and the joint pdf derivatives. */
      this->UpdateJointPDFAndDerivatives(
        fixedImageValue, movingImageValue, &imageJacobian, &nzji, this->m_JointPDF.GetPointer() );

    } //end if-block check sampleOk
  }   // end iterating over fixed image spatial sample container for loop

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Compute alpha. */
  this->m_Alpha = 0.0;
  if( this->m_NumberOfPixelsCounted > 0 )
  {
    this->m_Alpha = 1.0 / static_cast< double >( this->m_NumberOfPixelsCounted );
  }

} // end ComputePDFsAndPDFDerivatives()


/**
 * ************************ ComputePDFsAndIncrementalPDFs *******************
 */

template< class TFixedImage, class TMovingImage >
void
ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ComputePDFsAndIncrementalPDFs( const ParametersType & parameters ) const
{
  /** Initialize some variables. */
  this->m_JointPDF->FillBuffer( 0.0 );
  this->m_IncrementalJointPDFRight->FillBuffer( 0.0 );
  this->m_IncrementalJointPDFLeft->FillBuffer( 0.0 );
  this->m_Alpha = 0.0;
  this->m_PerturbedAlphaRight.Fill( 0.0 );
  this->m_PerturbedAlphaLeft.Fill( 0.0 );

  this->m_NumberOfPixelsCounted = 0;
  double       sumOfMovingMaskValues = 0.0;
  const double delta                 = this->GetFiniteDifferencePerturbation();

  /** sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  TransformJacobianType      jacobian;

  /** Arrays that store dM(x)/dmu and dMask(x)/dmu. */
  DerivativeType movingImageValuesRight( nzji.size() );
  DerivativeType movingImageValuesLeft( nzji.size() );
  DerivativeType movingMaskValuesRight( nzji.size() );
  DerivativeType movingMaskValuesLeft( nzji.size() );

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();

  /** Loop over sample container and compute contribution of each sample to pdfs. */
  for( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates. */
    const FixedImagePointType & fixedPoint = ( *fiter ).Value().m_ImageCoordinates;

    /** Transform point and check if it is inside the B-spline support region.
     * if not, skip this sample.
     */
    MovingImagePointType mappedPoint;
    bool                 sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    if( sampleOk )
    {
      /** Get the fixed image value and make sure the value falls within the histogram range. */
      RealType fixedImageValue = static_cast< RealType >( ( *fiter ).Value().m_ImageValue );
      fixedImageValue = this->GetFixedImageLimiter()->Evaluate( fixedImageValue );

      /** Check if point is inside mask. */
      sampleOk = this->IsInsideMovingMask( mappedPoint );
      RealType movingMaskValue
        = static_cast< RealType >( static_cast< unsigned char >( sampleOk ) );

      /** Compute the moving image value M(T(x)) and check if
      * the point is inside the moving image buffer.
      */
      RealType movingImageValue = itk::NumericTraits< RealType >::Zero;
      if( sampleOk )
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, 0 );
        if( sampleOk )
        {
          movingImageValue = this->GetMovingImageLimiter()->Evaluate( movingImageValue );
        }
        else
        {
          /** this movingImageValueRight is invalid, even though the mask indicated it is valid. */
          movingMaskValue = 0.0;
        }
      }

      /** Stop with this sample. It may be possible that with a perturbed parameter
       * a valid voxel pair is obtained, but:
       * - this chance is small,
       * - quitting now saves a lot of time, especially because this situation
       *   occurs at border pixels (there are a lot of those)
       * - if we would analytically compute the gradient the same choice is
       *   somehow made.
       */
      if( !sampleOk ) { continue; }

      /** Count how many samples were used. */
      sumOfMovingMaskValues         += movingMaskValue;
      this->m_NumberOfPixelsCounted += static_cast< unsigned int >( sampleOk );

      /** Get the TransformJacobian dT/dmu. We assume the transform is a linear
       * function of its parameters, so that we can evaluate T(x;\mu+delta_ek)
       * as T(x) + delta * dT/dmu_k.
       */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      MovingImagePointType mappedPointRight;
      MovingImagePointType mappedPointLeft;

      /** Loop over all parameters to perturb (parameters with nonzero Jacobian). */
      for( unsigned int i = 0; i < nzji.size(); ++i )
      {
        /** Compute the transformed input point after perturbation. */
        for( unsigned int j = 0; j < MovingImageDimension; ++j )
        {
          const double delta_jac = delta * jacobian[ j ][ i ];
          mappedPointRight[ j ] = mappedPoint[ j ] + delta_jac;
          mappedPointLeft[ j ]  = mappedPoint[ j ] - delta_jac;
        }

        /** Compute the moving mask 'value' and moving image value at the right perturbed positions. */
        sampleOk = this->IsInsideMovingMask( mappedPointRight );
        RealType movingMaskValueRight
          = static_cast< RealType >( static_cast< unsigned char >( sampleOk ) );
        if( sampleOk )
        {
          RealType movingImageValueRight = 0.0;
          sampleOk = this->EvaluateMovingImageValueAndDerivative(
            mappedPointRight, movingImageValueRight, 0 );
          if( sampleOk )
          {
            movingImageValueRight
                                        = this->GetMovingImageLimiter()->Evaluate( movingImageValueRight );
            movingImageValuesRight[ i ] = movingImageValueRight;
          }
          else
          {
            /** this movingImageValueRight is invalid, even though the mask indicated it is valid. */
            movingMaskValueRight = 0.0;
          }
        }
        movingMaskValuesRight[ i ] = movingMaskValueRight;

        /** Compute the moving mask and moving image value at the left perturbed positions. */
        sampleOk = this->IsInsideMovingMask( mappedPointLeft );
        RealType movingMaskValueLeft
          = static_cast< RealType >( static_cast< unsigned char >( sampleOk ) );
        if( sampleOk )
        {
          RealType movingImageValueLeft = 0.0;
          sampleOk = this->EvaluateMovingImageValueAndDerivative(
            mappedPointLeft, movingImageValueLeft, 0 );
          if( sampleOk )
          {
            movingImageValueLeft
                                       = this->GetMovingImageLimiter()->Evaluate( movingImageValueLeft );
            movingImageValuesLeft[ i ] = movingImageValueLeft;
          }
          else
          {
            /** this movingImageValueLeft is invalid, even though the mask indicated it is valid. */
            movingMaskValueLeft = 0.0;
          }
        }
        movingMaskValuesLeft[ i ] = movingMaskValueLeft;

      } // next parameter to perturb

      /** Update the joint pdf and the incremental joint pdfs, and the
       * perturbed alpha arrays.
       */
      this->UpdateJointPDFAndIncrementalPDFs(
        fixedImageValue, movingImageValue, movingMaskValue,
        movingImageValuesRight, movingImageValuesLeft,
        movingMaskValuesRight, movingMaskValuesLeft, nzji );

    } //end if-block check sampleOk
  }   // end iterating over fixed image spatial sample container for loop

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Compute alpha and its perturbed versions. */
  this->m_Alpha = 0.0;
  if( sumOfMovingMaskValues > 1e-14 )
  {
    this->m_Alpha = 1.0 / sumOfMovingMaskValues;
  }
  for( unsigned int i = 0; i < this->GetNumberOfParameters(); ++i )
  {
    this->m_PerturbedAlphaRight[ i ] += sumOfMovingMaskValues;
    this->m_PerturbedAlphaLeft[ i ]  += sumOfMovingMaskValues;
    if( this->m_PerturbedAlphaRight[ i ] > 1e-10 )
    {
      this->m_PerturbedAlphaRight[ i ] = 1.0 / this->m_PerturbedAlphaRight[ i ];
    }
    else
    {
      this->m_PerturbedAlphaRight[ i ] = 0.0;
    }
    if( this->m_PerturbedAlphaLeft[ i ] > 1e-10 )
    {
      this->m_PerturbedAlphaLeft[ i ] = 1.0 / this->m_PerturbedAlphaLeft[ i ];
    }
    else
    {
      this->m_PerturbedAlphaLeft[ i ] = 0.0;
    }
  }

} // end ComputePDFsAndIncrementalPDFs()


} // end namespace itk

#endif // end #ifndef _itkParzenWindowHistogramImageToImageMetric_HXX__
