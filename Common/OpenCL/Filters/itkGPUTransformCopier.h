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
#ifndef itkGPUTransformCopier_h
#define itkGPUTransformCopier_h

#include "itkTransform.h"

namespace itk
{
/** \class GPUTransformCopier
 * \brief A helper class which creates an GPU transform which
 * is perfect copy of the CPU transform.
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
 *  typedef itk::GPUTransformCopier< OCLImageTypes, OCLImageDims, TransformType, float > CopierType;
 *  CopierType::Pointer copier = CopierType::New();
 *  copier->SetInputTransform(CPUTransform);
 *  copier->Update();
 *  TransformType::Pointer GPUTransform = copier->GetModifiableOutput();
 * \endcode
 *
 * Note that the Update() method must be called explicitly in the filter
 * that provides the input to the GPUTransformCopier object. This is needed
 * because the GPUTransformCopier is not a pipeline filter.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType>
class ITK_TEMPLATE_EXPORT GPUTransformCopier : public Object
{
public:
  /** Standard class typedefs. */
  typedef GPUTransformCopier       Self;
  typedef Object                   Superclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUTransformCopier, Object);

  /** Type CPU definitions for the transform. */
  typedef TTransform                                     CPUTransformType;
  typedef typename CPUTransformType::ConstPointer        CPUTransformConstPointer;
  typedef typename CPUTransformType::ParametersType      CPUParametersType;
  typedef typename CPUTransformType::FixedParametersType CPUFixedParametersType;
  typedef typename CPUTransformType::ScalarType          CPUScalarType;

  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, CPUTransformType::InputSpaceDimension);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, CPUTransformType::OutputSpaceDimension);

  /** Type GPU definitions for the transform. */
  typedef TOutputTransformPrecisionType                                       GPUScalarType;
  typedef Transform<GPUScalarType, InputSpaceDimension, OutputSpaceDimension> GPUTransformType;
  typedef typename GPUTransformType::Pointer                                  GPUTransformPointer;
  typedef typename GPUTransformType::ParametersType                           GPUParametersType;
  typedef typename GPUTransformType::FixedParametersType                      GPUFixedParametersType;

  /** Get/Set the input transform. */
  itkSetConstObjectMacro(InputTransform, CPUTransformType);

  /** Compute of the output transform. */
  itkGetModifiableObjectMacro(Output, GPUTransformType);

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
  GPUTransformCopier();
  ~GPUTransformCopier() override {}
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Method to copy the transforms parameters. */
  bool
  CopyTransform(const CPUTransformConstPointer & fromTransform, GPUTransformPointer & toTransform);

  /** Cast and copy the transform parameters. */
  void
  CastCopyTransformParameters(const CPUTransformConstPointer & fromTransform, GPUTransformPointer & toTransform);

  /** Method to copy the parameters. */
  void
  CastCopyParameters(const CPUParametersType & from, GPUParametersType & to);

  /** Method to copy the fixed parameters. */
  void
  CastCopyFixedParameters(const CPUFixedParametersType & from, GPUFixedParametersType & to);

private:
  /** Copy method for BSpline transform. */
  bool
  CopyBSplineTransform(const CPUTransformConstPointer & fromTransform, GPUTransformPointer & toTransform);

  /** Templated struct to capture the transform space dimension */
  template <unsigned int Dimension>
  struct ITK_TEMPLATE_EXPORT TransformSpaceDimensionToType
  {};

  /** Copy method for Euler2D transform. */
  template <unsigned int InputSpaceDimension>
  bool
  CopyEuler2DTransform(const CPUTransformConstPointer &,
                       GPUTransformPointer &,
                       TransformSpaceDimensionToType<InputSpaceDimension>)
  {
    return false;
  }


  /** Copy method for Euler3D transform. */
  template <unsigned int InputSpaceDimension>
  bool
  CopyEuler3DTransform(const CPUTransformConstPointer &,
                       GPUTransformPointer &,
                       TransformSpaceDimensionToType<InputSpaceDimension>)
  {
    return false;
  }


  /** Copy method for Euler2D transform, partial specialization. */
  bool
  CopyEuler2DTransform(const CPUTransformConstPointer & fromTransform,
                       GPUTransformPointer &            toTransform,
                       TransformSpaceDimensionToType<2>);

  /** Copy method for Euler3D transform, partial specialization. */
  bool
  CopyEuler3DTransform(const CPUTransformConstPointer & fromTransform,
                       GPUTransformPointer &            toTransform,
                       TransformSpaceDimensionToType<3>);

  /** Copy method for Similarity2D transform. */
  template <unsigned int InputSpaceDimension>
  bool
  CopySimilarity2DTransform(const CPUTransformConstPointer &,
                            GPUTransformPointer &,
                            TransformSpaceDimensionToType<InputSpaceDimension>)
  {
    return false;
  }


  /** Copy method for Similarity3D transform. */
  template <unsigned int InputSpaceDimension>
  bool
  CopySimilarity3DTransform(const CPUTransformConstPointer &,
                            GPUTransformPointer &,
                            TransformSpaceDimensionToType<InputSpaceDimension>)
  {
    return false;
  }


  /** Copy method for Similarity2D transform, partial specialization. */
  bool
  CopySimilarity2DTransform(const CPUTransformConstPointer & fromTransform,
                            GPUTransformPointer &            toTransform,
                            TransformSpaceDimensionToType<2>);

  /** Copy method for Similarity3D transform, partial specialization. */
  bool
  CopySimilarity3DTransform(const CPUTransformConstPointer & fromTransform,
                            GPUTransformPointer &            toTransform,
                            TransformSpaceDimensionToType<3>);

private:
  GPUTransformCopier(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  CPUTransformConstPointer m_InputTransform;
  GPUTransformPointer      m_Output;
  ModifiedTimeType         m_InternalTransformTime;
  bool                     m_ExplicitMode;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUTransformCopier.hxx"
#endif

#endif /* itkGPUTransformCopier_h */
