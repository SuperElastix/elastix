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
#ifndef __elxElastixTemplate_h
#define __elxElastixTemplate_h

#include "elxElastixBase.h"
#include "itkObject.h"

#include "itkObjectFactory.h"
#include "itkCommand.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageToImageMetric.h"

#include "elxRegistrationBase.h"
#include "elxFixedImagePyramidBase.h"
#include "elxMovingImagePyramidBase.h"
#include "elxInterpolatorBase.h"
#include "elxImageSamplerBase.h"
#include "elxMetricBase.h"
#include "elxOptimizerBase.h"
#include "elxResamplerBase.h"
#include "elxResampleInterpolatorBase.h"
#include "elxTransformBase.h"

#include "itkTimeProbe.h"

#include <sstream>
#include <fstream>

/**
 * Macro that defines to functions. In the case of
 *   _name = Metric and _elxBaseType = MetricBaseType
 * this results in:
 * MetricBaseType * GetElxMetricBase(void) const;
 * MetricBaseType * GetElxMetricBase(unsigned int idx) const;
 *
 * The first function simply calls the second with argument = 0.
 * The second retrieves the metric component from the MetricContainer
 * and casts it to a MetricBaseType*;
 *
 * This macro is #undef'ed at the end of this header file.
 */

#define elxGetBaseMacro( _name, _elxbasetype ) \
  virtual _elxbasetype * GetElx##_name##Base( void ) const \
  { \
    return this->GetElx##_name##Base( 0 ); \
  } \
  _elxbasetype * GetElx##_name##Base( unsigned int idx ) const \
  { \
    if( idx < this->GetNumberOf##_name##s() ) \
    { \
      return dynamic_cast< _elxbasetype * >( \
        this->Get##_name##Container()->ElementAt( idx ).GetPointer() ); \
    } \
    return 0; \
  }
//end elxGetBaseMacro

namespace elastix
{

/**
 * \class ElastixTemplate
 * \brief The main elastix class, which connects components
 * and invokes the BeforeRegistration(), BeforeEachResolution(),
 * etc. methods.
 *
 * The ElastixTemplate class ...
 *
 * \parameter WriteTransformParametersEachIteration: Controls whether
 *    to save a transform parameter file to disk in every iteration.\n
 *    example: <tt>(WriteTransformParametersEachIteration "true")</tt>\n
 *    This parameter can not be specified for each resolution separately.
 *    Default value: "false".
 * \parameter WriteTransformParametersEachResolution: Controls whether
 *    to save a transform parameter file to disk in every resolution.\n
 *    example: <tt>(WriteTransformParametersEachResolution "true")</tt>\n
 *    This parameter can not be specified for each resolution separately.
 *    Default value: "false".
 * \parameter UseDirectionCosines: Controls whether to use or ignore the
 * direction cosines (world matrix, transform matrix) set in the images.
 * Voxel spacing and image origin are always taken into account, regardless
 * the setting of this parameter.\n
 *    example: <tt>(UseDirectionCosines "true")</tt>\n
 * Default: true. Recommended: true. This parameter was introduced in
 * elastix 4.3, with a default value of false for backward compabitility.
 *  From elastix 4.8 the default value has been changed to true. Setting it
 *  to false means that you choose to ignore important information from the
 *  image, which relates voxel coordinates to world coordinates. Ignoring it
 *  may easily lead to left/right swaps for example, which could skrew up a
 *  (medical) analysis.
 *
 * \ingroup Kernel
 */

template< class TFixedImage, class TMovingImage >
class ElastixTemplate : public itk::Object, public ElastixBase
{
public:

  /** Standard itk. */
  typedef ElastixTemplate                 Self;
  typedef itk::Object                     Superclass1;
  typedef ElastixBase                     Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ElastixTemplate, itk::Object );

  /** Typedefs inherited from Superclass2. */
  typedef Superclass2::ConfigurationType          ConfigurationType;
  typedef Superclass2::ConfigurationPointer       ConfigurationPointer;
  typedef Superclass2::ObjectType                 ObjectType;                   //for the components
  typedef Superclass2::DataObjectType             DataObjectType;               //for the images
  typedef Superclass2::ObjectPointer              ObjectPointer;
  typedef Superclass2::DataObjectPointer          DataObjectPointer;
  typedef Superclass2::ObjectContainerType        ObjectContainerType;
  typedef Superclass2::DataObjectContainerType    DataObjectContainerType;
  typedef Superclass2::FileNameContainerType      FileNameContainerType;
  typedef Superclass2::ObjectContainerPointer     ObjectContainerPointer;
  typedef Superclass2::DataObjectContainerPointer DataObjectContainerPointer;
  typedef Superclass2::FileNameContainerPointer   FileNameContainerPointer;

  /** Typedef's for this class. */
  typedef TFixedImage                       FixedImageType;
  typedef TMovingImage                      MovingImageType;
  typedef typename FixedImageType::Pointer  FixedImagePointer;
  typedef typename MovingImageType::Pointer MovingImagePointer;

  /** Result image */
  typedef itk::DataObject ResultImageType;

  /** Result deformation field */
  typedef itk::DataObject ResultDeformationFieldType;

  /** For using the Dimensions. */
  itkStaticConstMacro( Dimension,       unsigned int, FixedImageType::ImageDimension );
  itkStaticConstMacro( FixedDimension,  unsigned int, FixedImageType::ImageDimension );
  itkStaticConstMacro( MovingDimension, unsigned int, MovingImageType::ImageDimension );

  /** Types for the masks. */
  typedef unsigned char                                MaskPixelType;
  typedef itk::Image< MaskPixelType, FixedDimension >  FixedMaskType;
  typedef itk::Image< MaskPixelType, MovingDimension > MovingMaskType;
  typedef typename FixedMaskType::Pointer              FixedMaskPointer;
  typedef typename MovingMaskType::Pointer             MovingMaskPointer;

  /** Typedef for the UseDirectionCosines option. */
  typedef typename FixedImageType::DirectionType FixedImageDirectionType;

  /** Type for representation of the transform coordinates. */
  typedef itk::CostFunction::ParametersValueType CoordRepType;   // double

  /** BaseComponent. */
  typedef BaseComponent BaseComponentType;

  /** A Pointer to a member function of a BaseComponentType. */
  typedef void (BaseComponentType::* PtrToMemberFunction)( void );
  typedef int (BaseComponentType::*  PtrToMemberFunction2)( void );

  /** Commands that react on Events and call Self::Function(void). */
  typedef itk::SimpleMemberCommand< Self >                  BeforeEachResolutionCommandType;
  typedef itk::SimpleMemberCommand< Self >                  AfterEachResolutionCommandType;
  typedef itk::SimpleMemberCommand< Self >                  AfterEachIterationCommandType;
  typedef typename BeforeEachResolutionCommandType::Pointer BeforeEachResolutionCommandPointer;
  typedef typename AfterEachResolutionCommandType::Pointer  AfterEachResolutionCommandPointer;
  typedef typename AfterEachIterationCommandType::Pointer   AfterEachIterationCommandPointer;

  /** The elastix basecomponent types. */
  typedef FixedImagePyramidBase< Self >    FixedImagePyramidBaseType;
  typedef MovingImagePyramidBase< Self >   MovingImagePyramidBaseType;
  typedef InterpolatorBase< Self >         InterpolatorBaseType;
  typedef elx::ImageSamplerBase< Self >    ImageSamplerBaseType;
  typedef MetricBase< Self >               MetricBaseType;
  typedef OptimizerBase< Self >            OptimizerBaseType;
  typedef RegistrationBase< Self >         RegistrationBaseType;
  typedef ResamplerBase< Self >            ResamplerBaseType;
  typedef ResampleInterpolatorBase< Self > ResampleInterpolatorBaseType;
  typedef elx::TransformBase< Self >       TransformBaseType;

  /** Typedef's for Timer class. */
  typedef itk::TimeProbe TimerType;

  /** Typedef's for ApplyTransform.
   * \todo How useful is this? It is not consequently supported, since the
   * the input image is stored in the MovingImageContainer anyway.
   */
  typedef MovingImageType InputImageType;
  typedef MovingImageType OutputImageType;

  /** Typedef that is used in the elastix dll version. */
  typedef itk::ParameterMapInterface::ParameterMapType ParameterMapType;

  /** Functions to set/get pointers to the elastix components.
   * Get the components as pointers to elxBaseType.
   */
  elxGetBaseMacro( FixedImagePyramid, FixedImagePyramidBaseType );
  elxGetBaseMacro( MovingImagePyramid, MovingImagePyramidBaseType );
  elxGetBaseMacro( Interpolator, InterpolatorBaseType );
  elxGetBaseMacro( ImageSampler, ImageSamplerBaseType );
  elxGetBaseMacro( Metric, MetricBaseType );
  elxGetBaseMacro( Optimizer, OptimizerBaseType );
  elxGetBaseMacro( Registration, RegistrationBaseType );
  elxGetBaseMacro( Resampler, ResamplerBaseType );
  elxGetBaseMacro( ResampleInterpolator, ResampleInterpolatorBaseType );
  elxGetBaseMacro( Transform, TransformBaseType );

  /** Get pointers to the images. They are obtained from the
   * {Fixed,Moving}ImageContainer and casted to the appropriate type.
   */
  virtual FixedImageType * GetFixedImage( void ) const
  {
    return this->GetFixedImage( 0 );
  }


  virtual FixedImageType * GetFixedImage( unsigned int idx ) const;

  virtual MovingImageType * GetMovingImage( void ) const
  {
    return this->GetMovingImage( 0 );
  }


  virtual MovingImageType * GetMovingImage( unsigned int idx ) const;

  /** Get pointers to the masks. They are obtained from the
   * {Fixed,Moving}MaskContainer and casted to the appropriate type.
   */
  virtual FixedMaskType * GetFixedMask( void ) const
  {
    return this->GetFixedMask( 0 );
  }


  virtual FixedMaskType * GetFixedMask( unsigned int idx ) const;

  virtual MovingMaskType * GetMovingMask( void ) const
  {
    return this->GetMovingMask( 0 );
  }


  virtual MovingMaskType * GetMovingMask( unsigned int idx ) const;

  /** Get pointers to the result image. They are obtained from the
   * ResultImageContainer and casted to the appropriate type.
   */
  virtual ResultImageType * GetResultImage( void ) const
  {
    return this->GetResultImage( 0 );
  }


  virtual ResultImageType * GetResultImage( unsigned int idx ) const;

  virtual int SetResultImage( DataObjectPointer result_image );


  virtual ResultDeformationFieldType * GetResultDeformationField( void ) const
  {
    return this->GetResultDeformationField( 0 );
  }


  virtual ResultDeformationFieldType * GetResultDeformationField( unsigned int idx ) const;

  virtual int SetResultDeformationField( DataObjectPointer result_deformationfield );

  /** Main functions:
   * Run() for registration, and ApplyTransform() for just
   * applying a transform to an image.
   */
  virtual int Run( void );

  virtual int ApplyTransform( void );

  /** The Callback functions. */
  virtual int BeforeAll( void );

  virtual int BeforeAllTransformix( void );

  virtual void BeforeRegistration( void );

  virtual void BeforeEachResolution( void );

  virtual void AfterEachResolution( void );

  virtual void AfterEachIteration( void );

  virtual void AfterRegistration( void );

  /** Get the iteration number. */
  itkGetConstMacro( IterationCounter, unsigned int );

  /** Get the name of the current transform parameter file. */
  itkGetStringMacro( CurrentTransformParameterFileName );

  /** Set configuration vector. Library only. */
  virtual void SetConfigurations( std::vector< ConfigurationPointer > & configurations );

  /** Return configuration from vector of configurations. Library only. */
  virtual ConfigurationPointer GetConfiguration( const size_t index );

  virtual ConfigurationPointer GetConfiguration()
  {
    return Superclass2::GetConfiguration();
  }


  /** Get the original direction cosines of the fixed image. Returns
   * false if it failed to determine the original fixed image direction. In
   * that case the direction var is left unchanged. If no fixed image is
   * present, it tries to read it from the parameter file. */
  virtual bool GetOriginalFixedImageDirection( FixedImageDirectionType & direction ) const;

protected:

  ElastixTemplate();
  virtual ~ElastixTemplate(){}

  /** Classes that contain a function to load multiple images, given a filename container. */
  typedef typename Superclass2::MultipleImageLoader< FixedImageType >  FixedImageLoaderType;
  typedef typename Superclass2::MultipleImageLoader< MovingImageType > MovingImageLoaderType;
  typedef typename Superclass2::MultipleImageLoader< FixedMaskType >   FixedMaskLoaderType;
  typedef typename Superclass2::MultipleImageLoader< MovingMaskType >  MovingMaskLoaderType;

  /** CallBack commands. */
  BeforeEachResolutionCommandPointer m_BeforeEachResolutionCommand;
  AfterEachIterationCommandPointer   m_AfterEachIterationCommand;
  AfterEachResolutionCommandPointer  m_AfterEachResolutionCommand;

  /** Timers. */
  TimerType m_Timer0;
  TimerType m_IterationTimer;
  TimerType m_ResolutionTimer;

  /** Store the CurrentTransformParameterFileName. */
  std::string m_CurrentTransformParameterFileName;

  /** A vector of configuration objects, needed when transformix is used as library. */
  std::vector< ConfigurationPointer > m_Configurations;

  /** Count the number of iterations. */
  unsigned int m_IterationCounter;

  /** CreateTransformParameterFile. */
  virtual void CreateTransformParameterFile( const std::string FileName,
    const bool ToLog );

  /** CreateTransformParametersMap. */
  virtual void CreateTransformParametersMap( void );

  /** GetTransformParametersMap. */
  virtual ParameterMapType GetTransformParametersMap( void ) const;

  /** Stores transformation parameters map. */
  ParameterMapType m_TransformParametersMap;

  /** Open the IterationInfoFile, where the table with iteration info is written to. */
  virtual void OpenIterationInfoFile( void );

  std::ofstream m_IterationInfoFile;

  /** Used by the callback functions, BeforeEachResolution() etc.).
   * This method calls a function in each component, in the following order:
   * \li Registration
   * \li Transform
   * \li ImageSampler
   * \li Metric
   * \li Interpolator
   * \li Optimizer
   * \li FixedImagePyramid
   * \li MovingImagePyramid
   * \li ResampleInterpolator
   * \li Resampler
   */
  void CallInEachComponent( PtrToMemberFunction func );

  int CallInEachComponentInt( PtrToMemberFunction2 func );

  /** Call in each component SetElastix(This) and set its ComponentLabel
   * (for example "Metric1"). This makes sure that the component knows its
   * own function in the registration process.
   */
  virtual void ConfigureComponents( Self * This );

  /** Set the direction in the superclass' m_OriginalFixedImageDirection variable */
  virtual void SetOriginalFixedImageDirection( const FixedImageDirectionType & arg );

private:

  ElastixTemplate( const Self & ); // purposely not implemented
  void operator=( const Self & );  // purposely not implemented

};

} // end namespace elastix

#undef elxGetBaseMacro

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxElastixTemplate.hxx"
#endif

#endif // end #ifndef __elxElastixTemplate_h
