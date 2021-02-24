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
#ifndef itkGPUCompositeTransformCopier_h
#define itkGPUCompositeTransformCopier_h

#include "itkCompositeTransform.h"
#include "itkGPUTransformCopier.h"

namespace itk
{
/** \class GPUCompositeTransformCopier
 * \brief A helper class which creates an GPU composite transform which
 * is perfect copy of the CPU composite transform.
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
 *  typedef typelist::MakeTypeList< short, float >::Type OCLImageTypes;
 *  typedef itk::CompositeTransform< float, 3 > TransformType;
 *  typedef itk::GPUCompositeTransformCopier< OCLImageTypes, OCLImageDims, TransformType, float > CopierType;
 *  CopierType::Pointer copier = CopierType::New();
 *  copier->SetInputTransform(CPUTransform);
 *  copier->Update();
 *  TransformType::Pointer GPUTransform = copier->GetModifiableOutput();
 * \endcode
 *
 * Note that the Update() method must be called explicitly in the filter
 * that provides the input to the GPUCompositeTransformCopier object. This is needed
 * because the GPUCompositeTransformCopier is not a pipeline filter.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TTypeList,
          typename NDimensions,
          typename TCompositeTransform,
          typename TOutputTransformPrecisionType>
class ITK_TEMPLATE_EXPORT GPUCompositeTransformCopier : public Object
{
public:
  /** Standard class typedefs. */
  typedef GPUCompositeTransformCopier Self;
  typedef Object                      Superclass;
  typedef SmartPointer<Self>          Pointer;
  typedef SmartPointer<const Self>    ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUCompositeTransformCopier, Object);

  /** Type CPU definitions for the transform. */
  typedef TCompositeTransform                               CPUCompositeTransformType;
  typedef typename CPUCompositeTransformType::ConstPointer  CPUCompositeTransformConstPointer;
  typedef typename CPUCompositeTransformType::TransformType CPUTransformType;
  typedef typename CPUTransformType::Pointer                CPUTransformPointer;
  typedef typename CPUCompositeTransformType::ScalarType    CPUScalarType;

  /** Dimension of the domain space.
   * TCompositeTransform::InputDimension and TCompositeTransform::OutputDimension
   * are the same just pick the select one of them. */
  itkStaticConstMacro(SpaceDimension, unsigned int, CPUCompositeTransformType::InputDimension);

  /** Type GPU definitions for the transform. */
  typedef TOutputTransformPrecisionType                     GPUScalarType;
  typedef CompositeTransform<GPUScalarType, SpaceDimension> GPUCompositeTransformType;
  typedef typename GPUCompositeTransformType::Pointer       GPUCompositeTransformPointer;

  /** Type definitions for the transform copier. */
  typedef GPUTransformCopier<TTypeList, NDimensions, CPUTransformType, GPUScalarType> GPUTransformCopierType;
  typedef typename GPUTransformCopierType::Pointer                                    GPUTransformCopierPointer;
  typedef typename GPUTransformCopierType::GPUTransformPointer                        GPUOutputTransformPointer;

  /** Get/Set the input transform. */
  itkSetConstObjectMacro(InputTransform, CPUCompositeTransformType);

  /** Compute of the output transform. */
  itkGetModifiableObjectMacro(Output, GPUCompositeTransformType);

  /** Get/Set the explicit mode. The default is true.
   * If the explicit mode has been set to false that means that early in the
   * code the factories has been created.
   * ObjectFactoryBase::RegisterFactory( GPUAffineTransformFactory::New() );
   * ObjectFactoryBase::RegisterFactory( GPUTranslationTransformFactory::New() );
   * ObjectFactoryBase::RegisterFactory( GPUBSplineTransformFactory::New() );
   * ObjectFactoryBase::RegisterFactory( GPUEuler3DTransformFactory::New() );
   * ObjectFactoryBase::RegisterFactory( GPUSimilarity3DTransformFactory::New() ); */
  itkGetConstMacro(ExplicitMode, bool);
  itkSetMacro(ExplicitMode, bool);

  /** Update method. */
  void
  Update(void);

#ifdef ITK_USE_CONCEPT_CHECKING
  // Begin concept checking
  itkConceptMacro(OutputTransformPrecisionTypeIsFloatingPointCheck,
                  (Concept::IsFloatingPoint<TOutputTransformPrecisionType>));
  // End concept checking
#endif

protected:
  GPUCompositeTransformCopier();
  virtual ~GPUCompositeTransformCopier() {}
  virtual void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  GPUCompositeTransformCopier(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  CPUCompositeTransformConstPointer m_InputTransform;
  GPUCompositeTransformPointer      m_Output;
  ModifiedTimeType                  m_InternalTransformTime;
  bool                              m_ExplicitMode;
  GPUTransformCopierPointer         m_TransformCopier;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUCompositeTransformCopier.hxx"
#endif

#endif /* itkGPUCompositeTransformCopier_h */
