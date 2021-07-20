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
#ifndef itkGPUAdvancedCombinationTransformCopier_h
#define itkGPUAdvancedCombinationTransformCopier_h

#include "itkGPUAdvancedCombinationTransform.h"

namespace itk
{
/** \class GPUAdvancedCombinationTransformCopier
 * \brief A helper class which creates an GPU AdvancedCombinationTransform which
 * is perfect copy of the CPU AdvancedCombinationTransform.
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
 *  typedef itk::AdvancedCombinationTransform< float, 3 > TransformType;
 *  typedef itk::GPUAdvancedCombinationTransformCopier< OCLImageTypes, OCLImageDims, TransformType, float > CopierType;
 *  CopierType::Pointer copier = CopierType::New();
 *  copier->SetInputTransform(CPUTransform);
 *  copier->Update();
 *  TransformType::Pointer GPUTransform = copier->GetModifiableOutput();
 * \endcode
 *
 * Note that the Update() method must be called explicitly in the filter
 * that provides the input to the GPUAdvancedCombinationTransformCopier object. This is needed
 * because the GPUAdvancedCombinationTransformCopier is not a pipeline filter.
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
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
class ITK_TEMPLATE_EXPORT GPUAdvancedCombinationTransformCopier : public Object
{
public:
  /** Standard class typedefs. */
  typedef GPUAdvancedCombinationTransformCopier Self;
  typedef Object                                Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUAdvancedCombinationTransformCopier, Object);

  /** Type CPU definitions for the transform. */
  typedef TAdvancedCombinationTransform CPUComboTransformType;

  /** Input and Output space dimension. */
  itkStaticConstMacro(SpaceDimension, unsigned int, CPUComboTransformType::SpaceDimension);

  /** CPU combo transform class typedefs. */
  typedef typename CPUComboTransformType::ConstPointer                 CPUComboTransformConstPointer;
  typedef typename CPUComboTransformType::CurrentTransformType         CPUCurrentTransformType;
  typedef typename CPUComboTransformType::CurrentTransformPointer      CPUCurrentTransformPointer;
  typedef typename CPUComboTransformType::CurrentTransformConstPointer CPUCurrentTransformConstPointer;
  typedef typename CPUComboTransformType::InitialTransformType         CPUInitialTransformType;
  typedef typename CPUComboTransformType::InitialTransformPointer      CPUInitialTransformPointer;
  typedef typename CPUComboTransformType::InitialTransformConstPointer CPUInitialTransformConstPointer;
  typedef typename CPUComboTransformType::TransformType                TransformType;             // itk::Transform
  typedef typename CPUComboTransformType::TransformTypePointer         TransformTypePointer;      // itk::Transform
  typedef typename CPUComboTransformType::TransformTypeConstPointer    TransformTypeConstPointer; // itk::Transform
  typedef typename CPUComboTransformType::ScalarType                   CPUScalarType;

  /** CPU advanced transform class typedefs. */
  typedef AdvancedTransform<CPUScalarType, SpaceDimension, SpaceDimension> CPUAdvancedTransformType;
  typedef typename CPUAdvancedTransformType::ParametersType                CPUParametersType;
  typedef typename CPUAdvancedTransformType::FixedParametersType           CPUFixedParametersType;

  /** GPU combo transform class typedefs. */
  typedef TOutputTransformPrecisionType                                  GPUScalarType;
  typedef GPUAdvancedCombinationTransform<GPUScalarType, SpaceDimension> GPUComboTransformType;
  typedef typename GPUComboTransformType::Pointer                        GPUComboTransformPointer;

  /** GPU advanced transform class typedefs. */
  typedef AdvancedTransform<GPUScalarType, SpaceDimension, SpaceDimension> GPUAdvancedTransformType;
  typedef typename GPUAdvancedTransformType::Pointer                       GPUAdvancedTransformPointer;
  typedef typename GPUAdvancedTransformType::ParametersType                GPUParametersType;
  typedef typename GPUAdvancedTransformType::FixedParametersType           GPUFixedParametersType;

  /** Get/Set the input transform. */
  itkSetConstObjectMacro(InputTransform, CPUComboTransformType);

  /** Compute of the output transform. */
  itkGetModifiableObjectMacro(Output, GPUComboTransformType);

  /** Get/Set the explicit mode. The default is true.
   * If the explicit mode has been set to false that means that early in the
   * code the factories has been created.
   * ObjectFactoryBase::RegisterFactory( GPUAdvancedCombinationTransformFactory::New() );
   * ObjectFactoryBase::RegisterFactory( GPUAdvancedMatrixOffsetTransformBaseFactory::New() );
   * ObjectFactoryBase::RegisterFactory( GPUAdvancedTranslationTransformFactory::New() );
   * ObjectFactoryBase::RegisterFactory( GPUAdvancedBSplineDeformableTransformFactory::New() );
   * ObjectFactoryBase::RegisterFactory( GPUAdvancedSimilarity3DTransformFactory::New() ); */
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
  GPUAdvancedCombinationTransformCopier();
  ~GPUAdvancedCombinationTransformCopier() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Method to copy the transforms parameters. */
  bool
  CopyToCurrentTransform(const CPUCurrentTransformConstPointer & fromTransform, GPUComboTransformPointer & toTransform);

  /** Cast and copy the transform parameters. */
  void
  CastCopyTransformParameters(const CPUCurrentTransformConstPointer & fromTransform,
                              GPUAdvancedTransformPointer &           toTransform);

  /** Method to copy the parameters. */
  void
  CastCopyParameters(const CPUParametersType & from, GPUParametersType & to);

  /** Method to copy the fixed parameters. */
  void
  CastCopyFixedParameters(const CPUFixedParametersType & from, GPUFixedParametersType & to);

private:
  /** Copy method for BSpline transform. */
  bool
  CopyBSplineTransform(const CPUCurrentTransformConstPointer & fromTransform, GPUComboTransformPointer & toTransform);

  /** Templated struct to capture the transform space dimension */
  template <unsigned int Dimension>
  struct ITK_TEMPLATE_EXPORT TransformSpaceDimensionToType
  {};

  /** Copy method for Euler2D transform. */
  template <unsigned int InputSpaceDimension>
  bool
  CopyEuler2DTransform(const CPUCurrentTransformConstPointer &,
                       GPUComboTransformPointer &,
                       TransformSpaceDimensionToType<InputSpaceDimension>)
  {
    return false;
  }


  /** Copy method for Euler3D transform. */
  template <unsigned int InputSpaceDimension>
  bool
  CopyEuler3DTransform(const CPUCurrentTransformConstPointer &,
                       GPUComboTransformPointer &,
                       TransformSpaceDimensionToType<InputSpaceDimension>)
  {
    return false;
  }


  /** Copy method for Euler2D transform, partial specialization. */
  bool
  CopyEuler2DTransform(const CPUCurrentTransformConstPointer & fromTransform,
                       GPUComboTransformPointer &              toTransform,
                       TransformSpaceDimensionToType<2>);

  /** Copy method for Euler3D transform, partial specialization. */
  bool
  CopyEuler3DTransform(const CPUCurrentTransformConstPointer & fromTransform,
                       GPUComboTransformPointer &              toTransform,
                       TransformSpaceDimensionToType<3>);

  /** Copy method for Similarity2D transform. */
  template <unsigned int InputSpaceDimension>
  bool
  CopySimilarity2DTransform(const CPUCurrentTransformConstPointer &,
                            GPUComboTransformPointer &,
                            TransformSpaceDimensionToType<InputSpaceDimension>)
  {
    return false;
  }


  /** Copy method for Similarity3D transform. */
  template <unsigned int InputSpaceDimension>
  bool
  CopySimilarity3DTransform(const CPUCurrentTransformConstPointer &,
                            GPUComboTransformPointer &,
                            TransformSpaceDimensionToType<InputSpaceDimension>)
  {
    return false;
  }


  /** Copy method for Similarity2D transform, partial specialization. */
  bool
  CopySimilarity2DTransform(const CPUCurrentTransformConstPointer & fromTransform,
                            GPUComboTransformPointer &              toTransform,
                            TransformSpaceDimensionToType<2>);

  /** Copy method for Similarity3D transform, partial specialization. */
  bool
  CopySimilarity3DTransform(const CPUCurrentTransformConstPointer & fromTransform,
                            GPUComboTransformPointer &              toTransform,
                            TransformSpaceDimensionToType<3>);

private:
  GPUAdvancedCombinationTransformCopier(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  CPUComboTransformConstPointer m_InputTransform;
  GPUComboTransformPointer      m_Output;
  ModifiedTimeType              m_InternalTransformTime;
  bool                          m_ExplicitMode;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUAdvancedCombinationTransformCopier.hxx"
#endif

#endif /* itkGPUAdvancedCombinationTransformCopier_h */
