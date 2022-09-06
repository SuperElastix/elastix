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
#ifndef itkMultiInputMultiResolutionImageRegistrationMethodBase_h
#define itkMultiInputMultiResolutionImageRegistrationMethodBase_h

#include "itkMultiResolutionImageRegistrationMethod2.h"
#include "itkMultiInputImageToImageMetricBase.h"
#include <vector>

/** defines a method that calls the same method
 * with an extra 0 argument.
 */
#define itkSimpleSetMacro(_name, _type)                                                                                \
  virtual void Set##_name(_type _arg) { this->Set##_name(_arg, 0); }

#define elxOverrideSimpleSetMacro(_name, _type)                                                                        \
  void Set##_name(_type _arg) override { this->Set##_name(_arg, 0); }

/** defines for example: SetNumberOfInterpolators(). */
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
  virtual unsigned int GetNumberOf##_name##s() const { return this->m_##_name##s.size(); }

namespace itk
{

/** \class MultiInputMultiResolutionImageRegistrationMethodBase
 * \brief Base class for multi-resolution image registration methods
 *
 * This class is an extension of the itk class
 * MultiResolutionImageRegistrationMethod. It allows the use
 * of multiple metrics, multiple images,
 * multiple interpolators, and/or multiple image pyramids.
 *
 * You may also set an interpolator/fixedimage/etc to NULL, if you
 * happen to know that the corresponding metric is not an
 * ImageToImageMetric, but a regularizer for example (which does
 * not need an image.
 *
 *
 * \sa ImageRegistrationMethod
 * \sa MultiResolutionImageRegistrationMethod
 * \sa MultiResolutionImageRegistrationMethod2
 * \ingroup RegistrationFilters
 */

template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT MultiInputMultiResolutionImageRegistrationMethodBase
  : public MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MultiInputMultiResolutionImageRegistrationMethodBase);

  /** Standard class typedefs. */
  using Self = MultiInputMultiResolutionImageRegistrationMethodBase;
  using Superclass = MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiInputMultiResolutionImageRegistrationMethodBase, MultiResolutionImageRegistrationMethod2);

  /**  Superclass types */
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::FixedImageRegionPyramidType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImageConstPointer;

  using typename Superclass::MetricType;
  using typename Superclass::MetricPointer;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::InterpolatorType;
  using typename Superclass::InterpolatorPointer;
  using typename Superclass::OptimizerType;
  using OptimizerPointer = typename OptimizerType::Pointer;
  using typename Superclass::FixedImagePyramidType;
  using typename Superclass::FixedImagePyramidPointer;
  using typename Superclass::MovingImagePyramidType;
  using typename Superclass::MovingImagePyramidPointer;

  using typename Superclass::TransformOutputType;
  using typename Superclass::TransformOutputPointer;
  using typename Superclass::TransformOutputConstPointer;

  using typename Superclass::ParametersType;
  using typename Superclass::DataObjectPointer;

  using FixedImageRegionPyramidVectorType = std::vector<FixedImageRegionPyramidType>;

  /** Typedef's for the MultiInput part. */
  using MultiInputMetricType = MultiInputImageToImageMetricBase<FixedImageType, MovingImageType>;
  using MultiInputMetricPointer = typename MultiInputMetricType::Pointer;
  using FixedImageVectorType = typename MultiInputMetricType ::FixedImageVectorType;
  using FixedImageRegionVectorType = typename MultiInputMetricType ::FixedImageRegionVectorType;
  using MovingImageVectorType = typename MultiInputMetricType ::MovingImageVectorType;
  using InterpolatorVectorType = typename MultiInputMetricType ::InterpolatorVectorType;
  using FixedImageInterpolatorType = typename MultiInputMetricType ::FixedImageInterpolatorType;
  using FixedImageInterpolatorVectorType = typename MultiInputMetricType ::FixedImageInterpolatorVectorType;
  using FixedImagePyramidVectorType = std::vector<FixedImagePyramidPointer>;
  using MovingImagePyramidVectorType = std::vector<MovingImagePyramidPointer>;

  /** The following methods all have a similar pattern. The
   * SetFixedImage() just calls SetFixedImage(0).
   * SetFixedImage(0) also calls the Superclass::SetFixedImage(). This
   * is defined by the elxOverrideSimpleSetMacro.
   * GetFixedImage() just returns GetFixedImage(0) == Superclass::m_FixedImage.
   */

  /** Set/Get the Fixed image. */
  virtual void
  SetFixedImage(const FixedImageType * _arg, unsigned int pos);

  virtual const FixedImageType *
  GetFixedImage(unsigned int pos) const;

  const FixedImageType *
  GetFixedImage() const override
  {
    return this->GetFixedImage(0);
  }
  elxOverrideSimpleSetMacro(FixedImage, const FixedImageType *);
  itkSetNumberOfMacro(FixedImage);
  itkGetNumberOfMacro(FixedImage);

  /** Set/Get the Fixed image region. */
  virtual void
  SetFixedImageRegion(FixedImageRegionType _arg, unsigned int pos);

  virtual const FixedImageRegionType &
  GetFixedImageRegion(unsigned int pos) const;

  const FixedImageRegionType &
  GetFixedImageRegion() const override
  {
    return this->GetFixedImageRegion(0);
  }
  elxOverrideSimpleSetMacro(FixedImageRegion, const FixedImageRegionType);
  itkSetNumberOfMacro(FixedImageRegion);
  itkGetNumberOfMacro(FixedImageRegion);

  /** Set/Get the FixedImagePyramid. */
  virtual void
  SetFixedImagePyramid(FixedImagePyramidType * _arg, unsigned int pos);

  virtual FixedImagePyramidType *
  GetFixedImagePyramid(unsigned int pos) const;

  FixedImagePyramidType *
  GetFixedImagePyramid() override
  {
    return this->GetFixedImagePyramid(0);
  }
  elxOverrideSimpleSetMacro(FixedImagePyramid, FixedImagePyramidType *);
  itkSetNumberOfMacro(FixedImagePyramid);
  itkGetNumberOfMacro(FixedImagePyramid);

  /** Set/Get the Moving image. */
  virtual void
  SetMovingImage(const MovingImageType * _arg, unsigned int pos);

  virtual const MovingImageType *
  GetMovingImage(unsigned int pos) const;

  const MovingImageType *
  GetMovingImage() const override
  {
    return this->GetMovingImage(0);
  }
  elxOverrideSimpleSetMacro(MovingImage, const MovingImageType *);
  itkSetNumberOfMacro(MovingImage);
  itkGetNumberOfMacro(MovingImage);

  /** Set/Get the MovingImagePyramid. */
  virtual void
  SetMovingImagePyramid(MovingImagePyramidType * _arg, unsigned int pos);

  virtual MovingImagePyramidType *
  GetMovingImagePyramid(unsigned int pos) const;

  MovingImagePyramidType *
  GetMovingImagePyramid() override
  {
    return this->GetMovingImagePyramid(0);
  }
  elxOverrideSimpleSetMacro(MovingImagePyramid, MovingImagePyramidType *);
  itkSetNumberOfMacro(MovingImagePyramid);
  itkGetNumberOfMacro(MovingImagePyramid);

  /** Set/Get the Interpolator. */
  virtual void
  SetInterpolator(InterpolatorType * _arg, unsigned int pos);

  virtual InterpolatorType *
  GetInterpolator(unsigned int pos) const;

  InterpolatorType *
  GetInterpolator() override
  {
    return this->GetInterpolator(0);
  }
  elxOverrideSimpleSetMacro(Interpolator, InterpolatorType *);
  itkSetNumberOfMacro(Interpolator);
  itkGetNumberOfMacro(Interpolator);

  /** Set/Get the FixedImageInterpolator. */
  virtual void
  SetFixedImageInterpolator(FixedImageInterpolatorType * _arg, unsigned int pos);

  virtual FixedImageInterpolatorType *
  GetFixedImageInterpolator(unsigned int pos) const;

  virtual FixedImageInterpolatorType *
  GetFixedImageInterpolator()
  {
    return this->GetFixedImageInterpolator(0);
  }
  itkSimpleSetMacro(FixedImageInterpolator, FixedImageInterpolatorType *);
  itkSetNumberOfMacro(FixedImageInterpolator);
  itkGetNumberOfMacro(FixedImageInterpolator);

  /** Set a metric that takes multiple inputs. */
  void
  SetMetric(MetricType * _arg) override;

  /** Get a metric that takes multiple inputs. */
  itkGetModifiableObjectMacro(MultiInputMetric, MultiInputMetricType);

  /** Method to return the latest modified time of this object or
   * any of its cached ivars.
   */
  ModifiedTimeType
  GetMTime() const override;

protected:
  /** Constructor. */
  MultiInputMultiResolutionImageRegistrationMethodBase() = default;

  /** Destructor. */
  ~MultiInputMultiResolutionImageRegistrationMethodBase() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the registration.
   */
  void
  GenerateData() override;

  /** Initialize by setting the interconnects between the components.
   * This method is executed at every level of the pyramid with the
   * values corresponding to this resolution .
   */
  void
  Initialize() override;

  /** Compute the size of the fixed region for each level of the pyramid. */
  void
  PreparePyramids() override;

  /** Function called by PreparePyramids, which checks if the user input
   * regarding the image pyramids is ok.
   */
  virtual void
  CheckPyramids();

  /** Function called by Initialize, which checks if the user input is ok. */
  virtual void
  CheckOnInitialize();

  /** Containers for the pointers supplied by the user */
  FixedImageVectorType             m_FixedImages;
  MovingImageVectorType            m_MovingImages;
  FixedImageRegionVectorType       m_FixedImageRegions;
  FixedImagePyramidVectorType      m_FixedImagePyramids;
  MovingImagePyramidVectorType     m_MovingImagePyramids;
  InterpolatorVectorType           m_Interpolators;
  FixedImageInterpolatorVectorType m_FixedImageInterpolators;

  /** This vector is filled by the PreparePyramids function. */
  FixedImageRegionPyramidVectorType m_FixedImageRegionPyramids;

  /** Dummy image region */
  FixedImageRegionType m_NullFixedImageRegion;

private:
  MultiInputMetricPointer m_MultiInputMetric;
};

} // end namespace itk

#undef itkSetNumberOfMacro
#undef itkGetNumberOfMacro
#undef elxOverrideSimpleSetMacro

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMultiInputMultiResolutionImageRegistrationMethodBase.hxx"
#endif

#endif // end #ifndef itkMultiInputMultiResolutionImageRegistrationMethodBase_h
