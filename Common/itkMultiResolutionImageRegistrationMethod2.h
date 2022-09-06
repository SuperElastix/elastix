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

/** This class is a slight modification of the original ITK class:
 * MultiResolutionImageRegistrationMethod.
 * The original copyright message is pasted here, which includes also
 * the version information: */
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date: 2008-06-27 17:50:36 +0200 (Fri, 27 Jun 2008) $
  Version:   $Revision: 1728 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkMultiResolutionImageRegistrationMethod2_h
#define itkMultiResolutionImageRegistrationMethod2_h

#include "itkProcessObject.h"
#include "itkAdvancedImageToImageMetric.h"
#include "itkSingleValuedNonLinearOptimizer.h"
#include "itkMultiResolutionPyramidImageFilter.h"
#include "itkNumericTraits.h"
#include "itkDataObjectDecorator.h"

namespace itk
{

/** \class MultiResolutionImageRegistrationMethod2
 * \brief Base class for multi-resolution image registration methods
 *
 * This class is a slight modification of the original ITK class:
 * MultiResolutionImageRegistrationMethod. Compared to that one, some
 * variables are made protected, instead of private. Also, this class
 * makes less assumptions about the image pyramids.
 *
 * ---------------------------
 *
 * Original ITK documentation:
 *
 * This class provides a generic interface for multi-resolution
 * registration using components of the registration framework.
 * See documentation for ImageRegistrationMethod for a description
 * of the registration framework components.
 *
 * The registration process is initiated by method StartRegistration().
 * The user must set the parameters of each component before calling
 * this method.
 *
 * The number of resolution level to process can be set via
 * SetNumberOfLevels(). At each resolution level, the user specified
 * registration components are used to register downsampled version of the
 * images by computing the transform parameters that will map one image onto
 * the other image.
 *
 * The downsampled images are provided by user specified
 * MultiResolutionPyramidImageFilters. User must specify the schedule
 * for each pyramid externally prior to calling StartRegistration().
 *
 * \warning If there is discrepancy between the number of level requested
 * and a pyramid schedule. The pyramid schedule will be overridden
 * with a default one.
 *
 * Before each resolution level an IterationEvent is invoked providing an
 * opportunity for a user interface to change any of the components,
 * change component parameters, or stop the registration.
 *
 * This class is templated over the fixed image type and the moving image
 * type.
 *
 * \sa ImageRegistrationMethod
 * \ingroup RegistrationFilters
 */
template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT MultiResolutionImageRegistrationMethod2 : public ProcessObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MultiResolutionImageRegistrationMethod2);

  /** Standard class typedefs. */
  using Self = MultiResolutionImageRegistrationMethod2;
  using Superclass = ProcessObject;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiResolutionImageRegistrationMethod2, ProcessObject);

  /**  Type of the Fixed image. */
  using FixedImageType = TFixedImage;
  using FixedImageConstPointer = typename FixedImageType::ConstPointer;
  using FixedImageRegionType = typename FixedImageType::RegionType;
  using FixedImageRegionPyramidType = std::vector<FixedImageRegionType>;

  /**  Type of the Moving image. */
  using MovingImageType = TMovingImage;
  using MovingImageConstPointer = typename MovingImageType::ConstPointer;

  /**  Type of the metric. */
  using MetricType = AdvancedImageToImageMetric<FixedImageType, MovingImageType>;
  using MetricPointer = typename MetricType::Pointer;

  /**  Type of the Transform . */
  using TransformType = typename MetricType::AdvancedTransformType;
  using TransformPointer = typename TransformType::Pointer;

  /** Type for the output: Using Decorator pattern for enabling
   * the Transform to be passed in the data pipeline.
   */
  using TransformOutputType = DataObjectDecorator<TransformType>;
  using TransformOutputPointer = typename TransformOutputType::Pointer;
  using TransformOutputConstPointer = typename TransformOutputType::ConstPointer;

  /**  Type of the Interpolator. */
  using InterpolatorType = typename MetricType::InterpolatorType;
  using InterpolatorPointer = typename InterpolatorType::Pointer;

  /**  Type of the optimizer. */
  using OptimizerType = SingleValuedNonLinearOptimizer;

  /** Type of the Fixed image multiresolution pyramid. */
  using FixedImagePyramidType = MultiResolutionPyramidImageFilter<FixedImageType, FixedImageType>;
  using FixedImagePyramidPointer = typename FixedImagePyramidType::Pointer;

  /** Type of the moving image multiresolution pyramid. */
  using MovingImagePyramidType = MultiResolutionPyramidImageFilter<MovingImageType, MovingImageType>;
  using MovingImagePyramidPointer = typename MovingImagePyramidType::Pointer;

  /** Type of the Transformation parameters This is the same type used to
   *  represent the search space of the optimization algorithm.
   */
  using ParametersType = typename MetricType::TransformParametersType;

  /** Smart Pointer type to a DataObject. */
  using DataObjectPointer = typename DataObject::Pointer;

  /** Method that initiates the registration. */
  virtual void
  StartRegistration();

  /** Method to stop the registration. */
  virtual void
  StopRegistration();

  /** Set/Get the Fixed image. */
  itkSetConstObjectMacro(FixedImage, FixedImageType);
  itkGetConstObjectMacro(FixedImage, FixedImageType);

  /** Set/Get the Moving image. */
  itkSetConstObjectMacro(MovingImage, MovingImageType);
  itkGetConstObjectMacro(MovingImage, MovingImageType);

  /** Set/Get the Optimizer. */
  itkSetObjectMacro(Optimizer, OptimizerType);
  itkGetModifiableObjectMacro(Optimizer, OptimizerType);

  /** Set/Get the Metric. */
  itkSetObjectMacro(Metric, MetricType);
  itkGetModifiableObjectMacro(Metric, MetricType);

  /** Set/Get the Metric. */
  itkSetMacro(FixedImageRegion, FixedImageRegionType);
  itkGetConstReferenceMacro(FixedImageRegion, FixedImageRegionType);

  /** Set/Get the Transform. */
  itkSetObjectMacro(Transform, TransformType);
  itkGetModifiableObjectMacro(Transform, TransformType);

  /** Set/Get the Interpolator. */
  itkSetObjectMacro(Interpolator, InterpolatorType);
  itkGetModifiableObjectMacro(Interpolator, InterpolatorType);

  /** Set/Get the Fixed image pyramid. */
  itkSetObjectMacro(FixedImagePyramid, FixedImagePyramidType);
  itkGetModifiableObjectMacro(FixedImagePyramid, FixedImagePyramidType);

  /** Set/Get the Moving image pyramid. */
  itkSetObjectMacro(MovingImagePyramid, MovingImagePyramidType);
  itkGetModifiableObjectMacro(MovingImagePyramid, MovingImagePyramidType);

  /** Set/Get the number of multi-resolution levels. */
  itkSetClampMacro(NumberOfLevels, unsigned long, 1, NumericTraits<unsigned long>::max());
  itkGetConstMacro(NumberOfLevels, unsigned long);

  /** Get the current resolution level being processed. */
  itkGetConstMacro(CurrentLevel, unsigned long);

  /** Set/Get the initial transformation parameters. */
  itkSetMacro(InitialTransformParameters, ParametersType);
  itkGetConstReferenceMacro(InitialTransformParameters, ParametersType);

  /** Set/Get the initial transformation parameters of the next resolution
   * level to be processed. The default is the last set of parameters of
   * the last resolution level.
   */
  itkSetMacro(InitialTransformParametersOfNextLevel, ParametersType);
  itkGetConstReferenceMacro(InitialTransformParametersOfNextLevel, ParametersType);

  /** Get the last transformation parameters visited by
   * the optimizer.
   */
  itkGetConstReferenceMacro(LastTransformParameters, ParametersType);

  /** Returns the transform resulting from the registration process. */
  const TransformOutputType *
  GetOutput() const;

  /** Make a DataObject of the correct type to be used as the specified
   * output.
   */
  virtual DataObjectPointer
  MakeOutput(unsigned int idx);

  /** Method to return the latest modified time of this object or
   * any of its cached ivars.
   */
  ModifiedTimeType
  GetMTime() const override;

protected:
  /** Constructor. */
  MultiResolutionImageRegistrationMethod2();

  /** Destructor. */
  ~MultiResolutionImageRegistrationMethod2() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the registration.
   */
  void
  GenerateData() override;

  /** Initialize by setting the interconnects between the components.
      This method is executed at every level of the pyramid with the
      values corresponding to this resolution
   */
  virtual void
  Initialize();

  /** Compute the size of the fixed region for each level of the pyramid. */
  virtual void
  PreparePyramids();

  /** Set the current level to be processed. */
  itkSetMacro(CurrentLevel, unsigned long);

  /** The last transform parameters. Compared to the ITK class
   * itk::MultiResolutionImageRegistrationMethod these member variables
   * are made protected, so they can be accessed by children classes.
   */
  ParametersType m_LastTransformParameters;
  bool           m_Stop;

private:
  /** Member variables. */
  MetricPointer          m_Metric;
  OptimizerType::Pointer m_Optimizer;
  TransformPointer       m_Transform;
  InterpolatorPointer    m_Interpolator;

  ParametersType m_InitialTransformParameters;
  ParametersType m_InitialTransformParametersOfNextLevel;

  MovingImageConstPointer   m_MovingImage;
  FixedImageConstPointer    m_FixedImage;
  MovingImagePyramidPointer m_MovingImagePyramid;
  FixedImagePyramidPointer  m_FixedImagePyramid;

  FixedImageRegionType        m_FixedImageRegion;
  FixedImageRegionPyramidType m_FixedImageRegionPyramid;

  unsigned long m_NumberOfLevels;
  unsigned long m_CurrentLevel;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMultiResolutionImageRegistrationMethod2.hxx"
#endif

#endif // end #ifndef itkMultiResolutionImageRegistrationMethod2_h
