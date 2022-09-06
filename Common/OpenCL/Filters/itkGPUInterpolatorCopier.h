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
#ifndef itkGPUInterpolatorCopier_h
#define itkGPUInterpolatorCopier_h

#include "itkInterpolateImageFunction.h"
#include "itkGPUImage.h"

namespace itk
{
/** \class GPUInterpolatorCopier
 * \brief A helper class which creates an GPU interpolator which
 * is perfect copy of the CPU interpolator.
 *
 * This class is NOT a filter. Although it has an API similar to a filter, this class
 * is not intended to be used in a pipeline. Instead, the typical use will be like
 * it is illustrated in the following code:
 *
 * \code
 *  struct OCLImageDims
 *  {
 *   itkStaticConstMacro( Support1D, bool, true );
 *   itkStaticConstMacro( Support2D, bool, true );
 *   itkStaticConstMacro( Support3D, bool, true );
 *  };
 *
 *  using ImageType = itk::Image< short, 3 >;
 *  using OCLImageTypes = typelist::MakeTypeList< short, float >::Type;
 *  using InterpolatorType = itk::InterpolateImageFunction< ImageType, float >;
 *  using CopierType = itk::GPUInterpolatorCopier< OCLImageTypes, OCLImageDims, InterpolatorType, float >;
 *  auto copier = CopierType::New();
 *  copier->SetInputInterpolator(CPUInterpolator);
 *  copier->Update();
 *  TransformType::Pointer GPUInterpolator = copier->GetModifiableOutput();
 * \endcode
 *
 * Note that the Update() method must be called explicitly in the filter
 * that provides the input to the GPUInterpolatorCopier object. This is needed
 * because the GPUInterpolatorCopier is not a pipeline filter.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TTypeList, typename NDimensions, typename TInterpolator, typename TOutputCoordRep>
class ITK_TEMPLATE_EXPORT GPUInterpolatorCopier : public Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPUInterpolatorCopier);

  /** Standard class typedefs. */
  using Self = GPUInterpolatorCopier;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUInterpolatorCopier, Object);

  /** Type CPU definitions for the interpolator. */
  using CPUInterpolatorType = TInterpolator;
  using CPUInterpolatorConstPointer = typename CPUInterpolatorType::ConstPointer;
  using CPUInputImageType = typename CPUInterpolatorType::InputImageType;
  using CPUCoordRepType = typename CPUInterpolatorType::CoordRepType;
  using GPUCoordRepType = TOutputCoordRep;

  /** Typedef's for non explicit GPU interpolator definitions. */
  using GPUInterpolatorType = InterpolateImageFunction<CPUInputImageType, GPUCoordRepType>;
  using GPUInterpolatorPointer = typename GPUInterpolatorType::Pointer;
  using GPUInterpolatorConstPointer = typename GPUInterpolatorType::ConstPointer;

  /** Typedef's for explicit GPU interpolator definitions. */
  using CPUInputImagePixelType = typename CPUInputImageType::PixelType;
  using GPUInputImageType = itk::GPUImage<CPUInputImagePixelType, CPUInputImageType::ImageDimension>;
  using GPUExplicitInterpolatorType = InterpolateImageFunction<GPUInputImageType, GPUCoordRepType>;
  using GPUExplicitInterpolatorPointer = typename GPUExplicitInterpolatorType::Pointer;
  using GPUExplicitInterpolatorConstPointer = typename GPUExplicitInterpolatorType::ConstPointer;

  /** Get/Set the input interpolator. */
  itkSetConstObjectMacro(InputInterpolator, CPUInterpolatorType);

  /** Compute of the non explicit output interpolator. */
  itkGetModifiableObjectMacro(Output, GPUInterpolatorType);

  /** Compute of the explicit output interpolator.
   * This output should be used when ExplicitMode has been set to true. */
  itkGetModifiableObjectMacro(ExplicitOutput, GPUExplicitInterpolatorType);

  /** Get/Set the explicit mode. The default is true.
   * If the explicit mode has been set to false that means that early in the
   * code the factories has been created.
   * ObjectFactoryBase::RegisterFactory( GPUNearestNeighborInterpolateImageFunctionFactory::New() );
   * ObjectFactoryBase::RegisterFactory( GPULinearInterpolateImageFunctionFactory::New() );
   * ObjectFactoryBase::RegisterFactory( GPUBSplineInterpolateImageFunctionFactory::New() ); */
  itkGetConstMacro(ExplicitMode, bool);
  itkSetMacro(ExplicitMode, bool);

  /** Update method. */
  void
  Update();

#ifdef ITK_USE_CONCEPT_CHECKING
  // Begin concept checking
  itkConceptMacro(OutputCoordRepIsFloatingPointCheck, (Concept::IsFloatingPoint<TOutputCoordRep>));
  // End concept checking
#endif

protected:
  GPUInterpolatorCopier();
  ~GPUInterpolatorCopier() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  CPUInterpolatorConstPointer    m_InputInterpolator;
  GPUInterpolatorPointer         m_Output;
  GPUExplicitInterpolatorPointer m_ExplicitOutput;
  ModifiedTimeType               m_InternalTransformTime;
  bool                           m_ExplicitMode;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUInterpolatorCopier.hxx"
#endif

#endif /* itkGPUInterpolatorCopier_h */
