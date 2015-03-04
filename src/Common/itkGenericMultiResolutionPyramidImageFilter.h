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
#ifndef __itkGenericMultiResolutionPyramidImageFilter_h
#define __itkGenericMultiResolutionPyramidImageFilter_h

#include "itkMultiResolutionPyramidImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"

namespace itk
{
/** \class GenericMultiResolutionPyramidImageFilter
 * \brief Framework for creating images in a multi-resolution
 * pyramid
 *
 * GenericMultiResolutionPyramidImageFilter provides a generic framework to
 * to create a image pyramid according to a user defined multi-resolution
 * rescale and smoothing schedules.
 *
 * The multi-resolution rescale schedule is specified in terms of
 * shrink factors at each multi-resolution level for each dimension
 *
 * The rescale schedule is stored as an unsigned int matrix.
 * An element of the table can be access via the double bracket
 * notation: schedule[level][dimension]
 * For example:
 * 8 4 4
 * 4 4 2
 *
 * is a rescale schedule for two computation level. In the first (coarest)
 * level the image is reduce by a factor of 8 in the column dimension,
 * factor of 4 in the row dimension and factor of 4 in the slice dimension.
 * In the second level, the image is reduce by a factor of 4 in the column
 * dimension, 4 is the row dimension and 2 in the slice dimension.
 *
 * The method SetNumberOfLevels() set the number of
 * computation levels in the pyramid. This method will
 * allocate memory for the multi-resolution rescale schedule table.
 * This method generates defaults tables with the starting
 * shrink factor for all dimension set to 2^(NumberOfLevel - 1)
 * All factors are halved for all subsequent levels.
 * For example if the number of levels was set to 4, the default table is:
 * 8 8 8
 * 4 4 4
 * 2 2 2
 * 1 1 1
 *
 * The user can get a copy of the rescale schedule via GetRescaleSchedule()
 * They may make alteration and reset it using SetRescaleSchedule()
 *
 * To generate each output image, recursive Gaussian smoothing is performed
 * using a SmoothingRecursiveGaussianImageFilter.
 *
 * The user can make alteration on smoothing schedule via SetSmoothingSchedule()
 * For example, for 4 levels smoothing schedule would be:
 * 3 4 5
 * 2 2 2
 * 0 1 2
 * 0 0 0
 *
 * In the first level all sigma are set to the same value 2 across each axis.
 * Sigma is measured in the units of image spacing. Use different values along
 * each axis if you would like perform nonidentical smoothing (see level 1)
 * Although for the level 2 no smoothing will be performed because all sigma
 * values are equal zeros. For the last level 3 smoothing will be performed with
 * sigma 0 for x axis.
 *
 * The default smoothing schedule is derived from the rescale schedule, where
 * each element is computed as: 0.5 * rescale_factor * image_spacing.
 *
 * The user can get a copy of the schedule via GetSmoothingSchedule()
 *
 * The smoothed image is then downsampled using a ResampleImageFilter or
 * ShrinkImageFilter depending on SetUseShrinkImageFilter().
 *
 * When this filter is updated, NumberOfLevels outputs are produced.
 * The N'th output correspond to the N'th level of the pyramid.
 *
 * The user can influence whether or not rescale schedule or smoothing schedule
 * will be used via SetRescaleScheduleToUnity() and
 * SetSmoothingScheduleToZero() methods.
 *
 * The GenericMultiResolutionPyramidImageFilter provides direct control to
 * compute only single level of the pyramid via SetCurrentLevel() and
 * SetComputeOnlyForCurrentLevel() methods.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * This implementation was taken from elastix (http://elastix.isi.uu.nl/).
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \sa SmoothingRecursiveGaussianImageFilter
 * \sa ResampleImageFilter
 * \sa ShrinkImageFilter
 *
 * \ingroup PyramidImageFilter MultiThreaded Streamed
 * \ingroup ITKRegistrationCommon
 */
template< class TInputImage, class TOutputImage, class TPrecisionType = double >
class GenericMultiResolutionPyramidImageFilter :
  public MultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
{
public:

  /** Standard class typedefs. */
  typedef GenericMultiResolutionPyramidImageFilter Self;
  typedef MultiResolutionPyramidImageFilter<
    TInputImage, TOutputImage >           Superclass;
  typedef typename Superclass::Superclass SuperSuperclass;
  typedef SmartPointer< Self >            Pointer;
  typedef SmartPointer< const Self >      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GenericMultiResolutionPyramidImageFilter,
    MultiResolutionPyramidImageFilter );

  /** ImageDimension enumeration. */
  itkStaticConstMacro( ImageDimension, unsigned int,
    TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int,
    TOutputImage::ImageDimension );

  /** Inherit types from Superclass. */
  typedef typename Superclass::ScheduleType                   ScheduleType;
  typedef typename Superclass::InputImageType                 InputImageType;
  typedef typename Superclass::OutputImageType                OutputImageType;
  typedef typename Superclass::InputImagePointer              InputImagePointer;
  typedef typename Superclass::OutputImagePointer             OutputImagePointer;
  typedef typename Superclass::InputImageConstPointer         InputImageConstPointer;
  typedef typename Superclass::InputImageType::SpacingType    SpacingType;
  typedef typename InputImageType::PixelType                  PixelType;
  typedef typename NumericTraits< PixelType >::ScalarRealType ScalarRealType;

  /** SmoothingScheduleType typedef support. */
  typedef Array2D< ScalarRealType > SmoothingScheduleType;
  typedef ScheduleType              RescaleScheduleType;

  /** Define the type for the sigma array. */
  typedef FixedArray< ScalarRealType,
    itkGetStaticConstMacro( ImageDimension ) > SigmaArrayType;
  typedef SigmaArrayType RescaleFactorArrayType;

  /** Set a multi-resolution schedule. The input schedule must have only
   * ImageDimension number of columns and NumberOfLevels number of rows. For
   * each dimension, the shrink factor must be non-increasing with respect to
   * subsequent levels. This function will clamp shrink factors to satisfy
   * this condition. All shrink factors less than one will also be clamped
   * to the value of 1.
   */
  virtual void SetSchedule( const ScheduleType & schedule );

  /** Set a multi-resolution rescale schedule. The input schedule must have only
   * ImageDimension number of columns and NumberOfLevels number of rows. For
   * each dimension, the shrink factor must be non-increasing with respect to
   * subsequent levels. This function will clamp shrink factors to satisfy
   * this condition. All shrink factors less than one will also be clamped
   * to the value of 1.
   */
  virtual void SetRescaleSchedule( const RescaleScheduleType & schedule );

  /** Set a multi-resolution rescale schedule with ones. */
  virtual void SetRescaleScheduleToUnity( void );

  /** Get the multi-resolution rescale schedule. */
  const RescaleScheduleType & GetRescaleSchedule( void ) const
  {
    return this->m_Schedule;
  }


  /** Set a multi-resolution smoothing schedule. The input schedule must have only
   * ImageDimension number of columns and NumberOfLevels number of rows.
   * All sigmas less than 0 will also be clamped to the value of 0.
   */
  virtual void SetSmoothingSchedule( const SmoothingScheduleType & schedule );

  /** Set a multi-resolution rescale schedule with zeros. */
  virtual void SetSmoothingScheduleToZero( void );

  /** Get the multi-resolution smoothing schedule. */
  itkGetConstReferenceMacro( SmoothingSchedule, SmoothingScheduleType );

  /** Set the number of multi-resolution levels. The matrices containing the
   * schedule will be resized accordingly. The schedules are populated with
   * default values.
   */
  virtual void SetNumberOfLevels( unsigned int num );

  /** Set the current multi-resolution levels. The current level is clamped to
   * a total number of levels.
   */
  virtual void SetCurrentLevel( unsigned int level );

  /** Get the current multi-resolution level. */
  itkGetConstReferenceMacro( CurrentLevel, unsigned int );

  /** Set a control on whether a current level will be used. */
  virtual void SetComputeOnlyForCurrentLevel( const bool _arg );

  itkGetConstMacro( ComputeOnlyForCurrentLevel, bool );
  itkBooleanMacro( ComputeOnlyForCurrentLevel );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro( SameDimensionCheck,
    ( Concept::SameDimension< ImageDimension, OutputImageDimension > ) );
  itkConceptMacro( OutputHasNumericTraitsCheck,
    ( Concept::HasNumericTraits< typename TOutputImage::PixelType > ) );
  /** End concept checking */
#endif

protected:

  GenericMultiResolutionPyramidImageFilter();
  ~GenericMultiResolutionPyramidImageFilter() {}

  /** PrintSelf. */
  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** GenericMultiResolutionPyramidImageFilter may produce images which are of
   * different resolution and different pixel spacing than its input image.
   * As such, GenericMultiResolutionPyramidImageFilter needs to provide an
   * implementation for GenerateOutputInformation() in order to inform the
   * pipeline execution model. The original documentation of this method is
   * below. \sa ProcessObject::GenerateOutputInformaton().
   */
  virtual void GenerateOutputInformation( void );

  /** Given one output whose requested region has been set, this method sets
   * the requested region for the remaining output images. The original
   * documentation of this method is below. \sa
   * ProcessObject::GenerateOutputRequestedRegion().
   */
  virtual void GenerateOutputRequestedRegion( DataObject * output );

  /** Overwrite the Superclass implementation: no padding required. */
  virtual void GenerateInputRequestedRegion( void );

  /** Generate the output data. */
  virtual void GenerateData( void );

  /** Release the output data when the current level is used. */
  void ReleaseOutputs( void );

  SmoothingScheduleType m_SmoothingSchedule;
  unsigned int          m_CurrentLevel;
  bool                  m_ComputeOnlyForCurrentLevel;
  bool                  m_SmoothingScheduleDefined;

private:

  /** Typedef for smoother. Smooth always happens first, then only from
   * InputImageType to OutputImageType is possible.
   */
  typedef SmoothingRecursiveGaussianImageFilter<
    InputImageType, OutputImageType > SmootherType;

  /** Typedefs for shrinker or resample. If smoother has not been used, then
   * we have to use InputImageType to OutputImageType,
   * otherwise OutputImageType to OutputImageType.
   */
  typedef ImageToImageFilter< OutputImageType, OutputImageType >
    ImageToImageFilterSameTypes;
  typedef ImageToImageFilter< InputImageType, OutputImageType >
    ImageToImageFilterDifferentTypes;

  /** Smooth image at current level. Returns true if performed.
   * This method does not perform execution.
   */
  bool SetupSmoother( const unsigned int level,
    typename SmootherType::Pointer & smoother,
    const InputImageConstPointer & input );

  /** Shrink or Resample image at current level. Returns 1 or 2 if performed,
   * 0 otherwise. This method does not perform execution.
   */
  int SetupShrinkerOrResampler( const unsigned int level,
    typename SmootherType::Pointer & smoother,
    const bool sameType,
    const InputImageConstPointer & input,
    const OutputImagePointer & outputPtr,
    typename ImageToImageFilterSameTypes::Pointer & rescaleSameTypes,
    typename ImageToImageFilterDifferentTypes::Pointer & rescaleDifferentTypes );

  /** Defines Shrink or Resample filters. */
  void DefineShrinkerOrResampler(
    const bool sameType,
    const RescaleFactorArrayType & shrinkFactors,
    const OutputImagePointer & outputPtr,
    typename ImageToImageFilterSameTypes::Pointer & rescaleSameTypes,
    typename ImageToImageFilterDifferentTypes::Pointer & rescaleDifferentTypes );

  /** Initialize m_SmoothingSchedule to default values for backward compatibility. */
  void SetSmoothingScheduleToDefault( void );

  /** Checks whether we have to compute anything based on
   * m_ComputeOnlyForCurrentLevel and m_CurrentLevel.
   */
  bool ComputeForCurrentLevel( const unsigned int level ) const;

  /** Backward compatibility method to compute default sigma value. */
  double GetDefaultSigma( const unsigned int level,
    const unsigned int dim,
    const unsigned int * factors,
    const SpacingType & spacing ) const;

  /** Get sigmas from m_SmoothingSchedule for the level. */
  void GetSigma( const unsigned int level,
    SigmaArrayType & sigmaArray ) const;

  /** Get shrink factors from m_Schedule for the level. */
  void GetShrinkFactors( const unsigned int level,
    RescaleFactorArrayType & shrinkFactors ) const;

  /** Returns true if all elements of sigmaArray are zeros,
   * otherwise return false.
   */
  bool AreSigmasAllZeros( const SigmaArrayType & sigmaArray ) const;

  /** Returns true if all elements of rescaleFactorArray are ones,
   * otherwise return false.
   */
  bool AreRescaleFactorsAllOnes( const RescaleFactorArrayType & rescaleFactorArray ) const;

  /** Returns true if smooth has been used in pipeline, otherwise return false. */
  bool IsSmoothingUsed( void ) const;

  /** Returns true if rescale has been used in pipeline, otherwise return false. */
  bool IsRescaleUsed( void ) const;

private:

  GenericMultiResolutionPyramidImageFilter( const Self & ); // purposely not implemented
  void operator=( const Self & );                           // purposely not implemented

};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGenericMultiResolutionPyramidImageFilter.hxx"
#endif

#endif
