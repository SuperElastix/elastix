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
#ifndef itkMultiMetricMultiResolutionImageRegistrationMethod_h
#define itkMultiMetricMultiResolutionImageRegistrationMethod_h

#include "itkMultiResolutionImageRegistrationMethod2.h"
#include "itkCombinationImageToImageMetric.h"
#include <vector>

/** defines a method that calls the same method
 * with an extra 0 argument */
#define elxOverrideSimpleSetMacro(_name, _type)                                                                        \
  void Set##_name(_type _arg) override { this->Set##_name(_arg, 0); }

/** defines for example: SetNumberOfInterpolators() */
#define itkSetNumberOfMacro(_name)                                                                                     \
  virtual void SetNumberOf##_name##s(unsigned int _arg)                                                                \
  {                                                                                                                    \
    if (this->m_##_name##s.size() != _arg)                                                                             \
    {                                                                                                                  \
      this->m_##_name##s.resize(_arg);                                                                                 \
      this->Modified();                                                                                                \
    }                                                                                                                  \
  }

/** defines for example: GetNumberOfInterpolators() */
#define itkGetNumberOfMacro(_name)                                                                                     \
  virtual unsigned int GetNumberOf##_name##s(void) const { return this->m_##_name##s.size(); }

namespace itk
{

/** \class MultiMetricMultiResolutionImageRegistrationMethod
 * \brief Base class for multi-resolution image registration methods
 *
 * This class is an extension of the itk class
 * MultiResolutionImageRegistrationMethod. It allows the use
 * of multiple metrics, which are summed, multiple images,
 * multiple interpolators, and/or multiple image pyramids.
 *
 * Make sure the following is true:\n
 *   nrofmetrics >= nrofinterpolators >= nrofmovingpyramids >= nrofmovingimages\n
 *   nrofmetrics >= nroffixedpyramids >= nroffixedimages\n
 *   nroffixedregions == nroffixedimages\n
 *
 *   nrofinterpolators == nrofmetrics OR nrofinterpolators == 1\n
 *   nroffixedimages == nrofmetrics OR nroffixedimages == 1\n
 *   etc...
 *
 * You may also set an interpolator/fixedimage/etc to NULL, if you
 * happen to know that the corresponding metric is not an
 * ImageToImageMetric, but a regularizer for example (which does
 * not need an image.
 *
 *
 * \sa ImageRegistrationMethod
 * \sa MultiResolutionImageRegistrationMethod
 * \ingroup RegistrationFilters
 */

template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT MultiMetricMultiResolutionImageRegistrationMethod
  : public MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>
{
public:
  /** Standard class typedefs. */
  typedef MultiMetricMultiResolutionImageRegistrationMethod                  Self;
  typedef MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage> Superclass;
  typedef SmartPointer<Self>                                                 Pointer;
  typedef SmartPointer<const Self>                                           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiMetricMultiResolutionImageRegistrationMethod, MultiResolutionImageRegistrationMethod2);

  /**  Superclass types */
  typedef typename Superclass::FixedImageType          FixedImageType;
  typedef typename Superclass::FixedImageConstPointer  FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType    FixedImageRegionType;
  typedef typename Superclass::MovingImageType         MovingImageType;
  typedef typename Superclass::MovingImageConstPointer MovingImageConstPointer;

  typedef typename Superclass::MetricType                MetricType;
  typedef typename Superclass::MetricPointer             MetricPointer;
  typedef typename Superclass::TransformType             TransformType;
  typedef typename Superclass::TransformPointer          TransformPointer;
  typedef typename Superclass::InterpolatorType          InterpolatorType;
  typedef typename Superclass::InterpolatorPointer       InterpolatorPointer;
  typedef typename Superclass::OptimizerType             OptimizerType;
  typedef typename OptimizerType::Pointer                OptimizerPointer;
  typedef typename Superclass::FixedImagePyramidType     FixedImagePyramidType;
  typedef typename Superclass::FixedImagePyramidPointer  FixedImagePyramidPointer;
  typedef typename Superclass::MovingImagePyramidType    MovingImagePyramidType;
  typedef typename Superclass::MovingImagePyramidPointer MovingImagePyramidPointer;

  typedef typename Superclass::TransformOutputType         TransformOutputType;
  typedef typename Superclass::TransformOutputPointer      TransformOutputPointer;
  typedef typename Superclass::TransformOutputConstPointer TransformOutputConstPointer;

  typedef typename Superclass::ParametersType    ParametersType;
  typedef typename Superclass::DataObjectPointer DataObjectPointer;

  /** Extra typedefs */
  typedef CombinationImageToImageMetric<FixedImageType, MovingImageType> CombinationMetricType;
  typedef typename CombinationMetricType::Pointer                        CombinationMetricPointer;

  /** Unfortunately the StopRegistration method is not virtual and
   * the m_Stop member is private in the superclass. That's why
   * we provide the following function to interrupt registration.
   */
  virtual void
  StopMultiMetricRegistration(void)
  {
    this->m_Stop = true;
  }


  /** Set the Metric. Reimplement this method to check if
   * the metric is a combination metric.
   * GetMetric returns the combination metric.
   * By default, a combination metric is already set on constructing
   * this class.
   */
  void
  SetMetric(MetricType * _arg) override;

  /** Get the metric as a pointer to a combination metric type.
   * Use this method to setup the combination metric (set weights,
   * nrofmetrics, submetrics, etc.
   */
  virtual CombinationMetricType *
  GetCombinationMetric(void) const
  {
    return this->m_CombinationMetric.GetPointer();
  }


  /** The following methods all have a similar pattern. The
   * SetFixedImage() just calls SetFixedImage(0).
   * SetFixedImage(0) also calls the Superclass::SetFixedImage(). This
   * is defined by the elxOverrideSimpleSetMacro.
   * GetFixedImage() just returns GetFixedImage(0) == Superclass::m_FixedImage.
   */

  /** Set/Get the fixed image. */
  virtual void
  SetFixedImage(const FixedImageType * _arg, unsigned int pos);

  virtual const FixedImageType *
  GetFixedImage(unsigned int pos) const;

  const FixedImageType *
  GetFixedImage(void) const override
  {
    return this->GetFixedImage(0);
  }


  elxOverrideSimpleSetMacro(FixedImage, const FixedImageType *);
  itkSetNumberOfMacro(FixedImage);
  itkGetNumberOfMacro(FixedImage);

  /** Set/Get the moving image. */
  virtual void
  SetMovingImage(const MovingImageType * _arg, unsigned int pos);

  virtual const MovingImageType *
  GetMovingImage(unsigned int pos) const;

  const MovingImageType *
  GetMovingImage(void) const override
  {
    return this->GetMovingImage(0);
  }
  elxOverrideSimpleSetMacro(MovingImage, const MovingImageType *);
  itkSetNumberOfMacro(MovingImage);
  itkGetNumberOfMacro(MovingImage);

  /** Set/Get the fixed image region. */
  virtual void
  SetFixedImageRegion(FixedImageRegionType _arg, unsigned int pos);

  virtual const FixedImageRegionType &
  GetFixedImageRegion(unsigned int pos) const;

  const FixedImageRegionType &
  GetFixedImageRegion(void) const override
  {
    return this->GetFixedImageRegion(0);
  }
  elxOverrideSimpleSetMacro(FixedImageRegion, const FixedImageRegionType);
  itkSetNumberOfMacro(FixedImageRegion);
  itkGetNumberOfMacro(FixedImageRegion);

  /** Set/Get the interpolator. */
  virtual void
  SetInterpolator(InterpolatorType * _arg, unsigned int pos);

  virtual InterpolatorType *
  GetInterpolator(unsigned int pos) const;

  InterpolatorType *
  GetInterpolator(void) override
  {
    return this->GetInterpolator(0);
  }
  elxOverrideSimpleSetMacro(Interpolator, InterpolatorType *);
  itkSetNumberOfMacro(Interpolator);
  itkGetNumberOfMacro(Interpolator);

  /** Set/Get the FixedImagePyramid. */
  virtual void
  SetFixedImagePyramid(FixedImagePyramidType * _arg, unsigned int pos);

  virtual FixedImagePyramidType *
  GetFixedImagePyramid(unsigned int pos) const;

  FixedImagePyramidType *
  GetFixedImagePyramid(void) override
  {
    return this->GetFixedImagePyramid(0);
  }
  elxOverrideSimpleSetMacro(FixedImagePyramid, FixedImagePyramidType *);
  itkSetNumberOfMacro(FixedImagePyramid);
  itkGetNumberOfMacro(FixedImagePyramid);

  /** Set/Get the MovingImagePyramid. */
  virtual void
  SetMovingImagePyramid(MovingImagePyramidType * _arg, unsigned int pos);

  virtual MovingImagePyramidType *
  GetMovingImagePyramid(unsigned int pos) const;

  MovingImagePyramidType *
  GetMovingImagePyramid(void) override
  {
    return this->GetMovingImagePyramid(0);
  }
  elxOverrideSimpleSetMacro(MovingImagePyramid, MovingImagePyramidType *);
  itkSetNumberOfMacro(MovingImagePyramid);
  itkGetNumberOfMacro(MovingImagePyramid);

  /** Method to return the latest modified time of this object or
   * any of its cached ivars.
   */
  ModifiedTimeType
  GetMTime(void) const override;

  /** Get the last transformation parameters visited by
   * the optimizer. Return the member variable declared in this class,
   * and not that of the superclass (which is declared private).
   */
  const ParametersType &
  GetLastTransformParameters(void) const override
  {
    return this->m_LastTransformParameters;
  }


protected:
  MultiMetricMultiResolutionImageRegistrationMethod();
  ~MultiMetricMultiResolutionImageRegistrationMethod() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  typedef std::vector<FixedImageRegionType> FixedImageRegionPyramidType;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the registration.
   */
  void
  GenerateData(void) override;

  /** Initialize by setting the interconnects between the components.
   * This method is executed at every level of the pyramid with the
   * values corresponding to this resolution.
   */
  void
  Initialize(void) override;

  /** Compute the size of the fixed region for each level of the pyramid.
   * Actually we would like to override PreparePyramids, but this function
   * is not virtual...
   */
  virtual void
  PrepareAllPyramids(void);

  /** Function called by PrepareAllPyramids, which checks if the user input
   * regarding the image pyramids is ok.
   */
  virtual void
  CheckPyramids(void);

  /** Function called by Initialize, which checks if the user input
   * is ok. Called by Initialize().
   */
  virtual void
  CheckOnInitialize(void);

  /** Variables already defined in the superclass, but as private...  */
  bool           m_Stop;
  ParametersType m_LastTransformParameters;

  /** A shortcut to m_Metric of type CombinationMetricPointer. */
  CombinationMetricPointer m_CombinationMetric;

  /** Containers for the pointers supplied by the user. */
  std::vector<FixedImageConstPointer>    m_FixedImages;
  std::vector<MovingImageConstPointer>   m_MovingImages;
  std::vector<FixedImageRegionType>      m_FixedImageRegions;
  std::vector<FixedImagePyramidPointer>  m_FixedImagePyramids;
  std::vector<MovingImagePyramidPointer> m_MovingImagePyramids;
  std::vector<InterpolatorPointer>       m_Interpolators;

  /** This vector is filled by the PrepareAllPyramids function. */
  std::vector<FixedImageRegionPyramidType> m_FixedImageRegionPyramids;

  /** Dummy image region. */
  FixedImageRegionType m_NullFixedImageRegion;

private:
  MultiMetricMultiResolutionImageRegistrationMethod(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#undef itkSetNumberOfMacro
#undef itkGetNumberOfMacro
#undef elxOverrideSimpleSetMacro

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMultiMetricMultiResolutionImageRegistrationMethod.hxx"
#endif

#endif
