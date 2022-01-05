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
#ifndef itkGPUAdvancedCombinationTransformCopier_hxx
#define itkGPUAdvancedCombinationTransformCopier_hxx

#include "itkGPUAdvancedCombinationTransformCopier.h"

// elastix CPU transforms
#include "itkAdvancedMatrixOffsetTransformBase.h"
#include "itkAdvancedTranslationTransform.h"
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkAdvancedRigid2DTransform.h"
#include "itkAdvancedEuler3DTransform.h"
#include "itkAdvancedSimilarity2DTransform.h"
#include "itkAdvancedSimilarity3DTransform.h"

// elastix GPU transforms
#include "itkGPUAdvancedMatrixOffsetTransformBase.h"
#include "itkGPUAdvancedTranslationTransform.h"
#include "itkGPUAdvancedBSplineDeformableTransform.h"
#include "itkGPUAdvancedEuler2DTransform.h"
#include "itkGPUAdvancedEuler3DTransform.h"
#include "itkGPUAdvancedSimilarity2DTransform.h"
#include "itkGPUAdvancedSimilarity3DTransform.h"

// GPU factory include
#include "itkGPUImageFactory.h"

namespace itk
{
//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
GPUAdvancedCombinationTransformCopier<TTypeList,
                                      NDimensions,
                                      TAdvancedCombinationTransform,
                                      TOutputTransformPrecisionType>::GPUAdvancedCombinationTransformCopier()
{
  this->m_InputTransform = nullptr;
  this->m_Output = nullptr;
  this->m_InternalTransformTime = 0;
  this->m_ExplicitMode = true;
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
void
GPUAdvancedCombinationTransformCopier<TTypeList,
                                      NDimensions,
                                      TAdvancedCombinationTransform,
                                      TOutputTransformPrecisionType>::Update()
{
  if (!this->m_InputTransform)
  {
    itkExceptionMacro(<< "ERROR: m_InputTransform not set");
    return;
  }

  // Update only if the input AdvancedCombinationTransform has been modified
  const ModifiedTimeType t = this->m_InputTransform->GetMTime();
  if (t <= this->m_InternalTransformTime)
    return; // No need to update

  // Cache the timestamp
  this->m_InternalTransformTime = t;

  // Allocate the output GPU combo transform
  GPUComboTransformPointer comboTransformGPU = GPUComboTransformType::New();
  this->m_Output = comboTransformGPU;

  // Initialize the current transforms
  CPUCurrentTransformConstPointer currentTransformCPU;
  GPUComboTransformPointer        currentTransformGPU = comboTransformGPU;

  // Loop over all sub-transforms
  const SizeValueType numberOfTransforms = this->m_InputTransform->GetNumberOfTransforms();
  for (SizeValueType i = 0; i < numberOfTransforms; ++i)
  {
    // Get the current CPU transform of type itk::Transform
    TransformTypePointer itkCurrentTransform = this->m_InputTransform->GetNthTransform(i);

    // Cast to advanced transform type, no checking needed
    currentTransformCPU = dynamic_cast<const CPUCurrentTransformType *>(itkCurrentTransform.GetPointer());

    // Copy the current CPU transform to the current GPU transform
    const bool copySucceeded = this->CopyToCurrentTransform(currentTransformCPU, currentTransformGPU);
    if (!copySucceeded)
    {
      itkExceptionMacro(<< "ERROR: GPUAdvancedCombinationTransformCopier was unable to copy transform from: "
                        << this->m_InputTransform);
    }

    // skip next step when last transform
    if (i == numberOfTransforms - 1)
    {
      continue;
    }

    // Reset the GPU combo transform
    GPUComboTransformPointer initialNext = GPUComboTransformType::New();
    currentTransformGPU->SetInitialTransform(initialNext);
    currentTransformGPU = initialNext;
  }
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
bool
GPUAdvancedCombinationTransformCopier<
  TTypeList,
  NDimensions,
  TAdvancedCombinationTransform,
  TOutputTransformPrecisionType>::CopyToCurrentTransform(const CPUCurrentTransformConstPointer & fromTransform,
                                                         GPUComboTransformPointer &              toTransform)
{
  if (fromTransform.IsNotNull())
  {
    // For Euler and Similarity transforms we have to use partial
    // template specialization logic.
    const unsigned int                                  InputDimension = SpaceDimension;
    const TransformSpaceDimensionToType<SpaceDimension> idim = {};

    // Try Advanced Euler
    bool eulerCopyResult = false;
    switch (InputDimension)
    {
      case 2:
        eulerCopyResult = this->CopyEuler2DTransform(fromTransform, toTransform, idim);
        break;
      case 3:
        eulerCopyResult = this->CopyEuler3DTransform(fromTransform, toTransform, idim);
        break;
      default:
        break;
    }

    if (eulerCopyResult)
    {
      return eulerCopyResult;
    }

    // Try Advanced Similarity
    bool similarityCopyResult = false;
    switch (InputDimension)
    {
      case 2:
        similarityCopyResult = this->CopySimilarity2DTransform(fromTransform, toTransform, idim);
        break;
      case 3:
        similarityCopyResult = this->CopySimilarity3DTransform(fromTransform, toTransform, idim);
        break;
      default:
        break;
    }

    if (similarityCopyResult)
    {
      return similarityCopyResult;
    }

    // Try Advanced Affine
    using AdvancedAffineTransformType =
      AdvancedMatrixOffsetTransformBase<CPUScalarType, SpaceDimension, SpaceDimension>;
    const typename AdvancedAffineTransformType::ConstPointer affine =
      dynamic_cast<const AdvancedAffineTransformType *>(fromTransform.GetPointer());

    if (affine)
    {
      GPUAdvancedTransformPointer affineTransform;
      if (this->m_ExplicitMode)
      {
        // Create GPU Advanced Affine transform in explicit mode
        using GPUAdvancedAffineTransformType = GPUAdvancedMatrixOffsetTransformBase<GPUScalarType, SpaceDimension>;
        affineTransform = GPUAdvancedAffineTransformType::New();
      }
      else
      {
        // Create GPU Advanced Affine transform in implicit mode
        using GPUAdvancedAffineTransformType =
          AdvancedMatrixOffsetTransformBase<GPUScalarType, SpaceDimension, SpaceDimension>;
        affineTransform = GPUAdvancedAffineTransformType::New();
      }
      this->CastCopyTransformParameters(fromTransform, affineTransform);
      toTransform->SetCurrentTransform(affineTransform);
      return true;
    }

    // Try Advanced Translation
    using AdvancedTranslationTransformType = AdvancedTranslationTransform<CPUScalarType, SpaceDimension>;
    const typename AdvancedTranslationTransformType::ConstPointer translation =
      dynamic_cast<const AdvancedTranslationTransformType *>(fromTransform.GetPointer());

    if (translation)
    {
      GPUAdvancedTransformPointer translationTransform;
      if (this->m_ExplicitMode)
      {
        // Create GPU Advanced Translation transform in explicit mode
        using GPUAdvancedTranslationTransformType = GPUAdvancedTranslationTransform<GPUScalarType, SpaceDimension>;
        translationTransform = GPUAdvancedTranslationTransformType::New();
      }
      else
      {
        // Create GPU Advanced Translation transform in implicit mode
        using GPUAdvancedTranslationTransformType = AdvancedTranslationTransform<GPUScalarType, SpaceDimension>;
        translationTransform = GPUAdvancedTranslationTransformType::New();
      }
      this->CastCopyTransformParameters(fromTransform, translationTransform);
      toTransform->SetCurrentTransform(translationTransform);
      return true;
    }

    // For BSpline we have to check all possible spline orders
    const bool bsplineCopyResult = this->CopyBSplineTransform(fromTransform, toTransform);
    if (bsplineCopyResult)
    {
      return bsplineCopyResult;
    }

    return false;
  }

  return true;
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
void
GPUAdvancedCombinationTransformCopier<
  TTypeList,
  NDimensions,
  TAdvancedCombinationTransform,
  TOutputTransformPrecisionType>::CastCopyTransformParameters(const CPUCurrentTransformConstPointer & fromTransform,
                                                              GPUAdvancedTransformPointer &           toTransform)
{
  const CPUFixedParametersType & fixedParametersFrom = fromTransform->GetFixedParameters();
  const CPUParametersType &      parametersFrom = fromTransform->GetParameters();

  GPUFixedParametersType fixedParametersTo;
  GPUParametersType      parametersTo;

  this->CastCopyFixedParameters(fixedParametersFrom, fixedParametersTo);
  this->CastCopyParameters(parametersFrom, parametersTo);

  toTransform->SetFixedParameters(fixedParametersTo);
  toTransform->SetParameters(parametersTo);
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
void
GPUAdvancedCombinationTransformCopier<TTypeList,
                                      NDimensions,
                                      TAdvancedCombinationTransform,
                                      TOutputTransformPrecisionType>::CastCopyParameters(const CPUParametersType & from,
                                                                                         GPUParametersType &       to)
{
  if (from.GetSize() == 0)
  {
    return;
  }

  to.SetSize(from.GetSize());
  for (SizeValueType i = 0; i < from.GetSize(); ++i)
  {
    to[i] = static_cast<GPUScalarType>(from[i]);
  }
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
void
GPUAdvancedCombinationTransformCopier<
  TTypeList,
  NDimensions,
  TAdvancedCombinationTransform,
  TOutputTransformPrecisionType>::CastCopyFixedParameters(const CPUFixedParametersType & from,
                                                          GPUFixedParametersType &       to)
{
  if (from.GetSize() == 0)
  {
    return;
  }

  to.SetSize(from.GetSize());
  for (SizeValueType i = 0; i < from.GetSize(); ++i)
  {
    to[i] = static_cast<GPUScalarType>(from[i]);
  }
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
bool
GPUAdvancedCombinationTransformCopier<
  TTypeList,
  NDimensions,
  TAdvancedCombinationTransform,
  TOutputTransformPrecisionType>::CopyBSplineTransform(const CPUCurrentTransformConstPointer & fromTransform,
                                                       GPUComboTransformPointer &              toTransform)
{
  using AdvancedBSplineOrder0TransformType = AdvancedBSplineDeformableTransform<CPUScalarType, SpaceDimension, 0>;
  using AdvancedBSplineOrder1TransformType = AdvancedBSplineDeformableTransform<CPUScalarType, SpaceDimension, 1>;
  using AdvancedBSplineOrder2TransformType = AdvancedBSplineDeformableTransform<CPUScalarType, SpaceDimension, 2>;
  using AdvancedBSplineOrder3TransformType = AdvancedBSplineDeformableTransform<CPUScalarType, SpaceDimension, 3>;

  GPUAdvancedTransformPointer bsplineTransform;

  // When creating the GPUAdvancedBSplineDeformableTransform in explicit mode
  // We also have to register GPUImageFactory because
  // GPUAdvancedBSplineDeformableTransform using m_Coefficients as ITK images
  // inside the implementation, therefore we define GPUImageFactory pointer
  using GPUImageFactoryType = itk::GPUImageFactory2<TTypeList, NDimensions>;
  using GPUImageFactoryPointer = typename GPUImageFactoryType::Pointer;

  // Try BSpline Order 3 first
  const typename AdvancedBSplineOrder3TransformType::ConstPointer bsplineOrder3 =
    dynamic_cast<const AdvancedBSplineOrder3TransformType *>(fromTransform.GetPointer());

  if (bsplineOrder3)
  {
    if (this->m_ExplicitMode)
    {
      // Register image factory
      GPUImageFactoryPointer imageFactory = GPUImageFactoryType::New();
      itk::ObjectFactoryBase::RegisterFactory(imageFactory);

      // Create GPU Advanced BSpline transform in explicit mode
      using GPUBSplineTransformType = GPUAdvancedBSplineDeformableTransform<GPUScalarType, SpaceDimension, 3>;
      bsplineTransform = GPUBSplineTransformType::New();

      // UnRegister image factory
      itk::ObjectFactoryBase::UnRegisterFactory(imageFactory);
    }
    else
    {
      // Create GPU Advanced BSpline transform in implicit mode
      using GPUBSplineTransformType = AdvancedBSplineDeformableTransform<GPUScalarType, SpaceDimension, 3>;
      bsplineTransform = GPUBSplineTransformType::New();
    }
    this->CastCopyTransformParameters(fromTransform, bsplineTransform);
    toTransform->SetCurrentTransform(bsplineTransform);
    return true;
  }
  else
  {
    // Try BSpline Order 0
    const typename AdvancedBSplineOrder0TransformType::ConstPointer bsplineOrder0 =
      dynamic_cast<const AdvancedBSplineOrder0TransformType *>(fromTransform.GetPointer());

    if (bsplineOrder0)
    {
      if (this->m_ExplicitMode)
      {
        // Register image factory
        GPUImageFactoryPointer imageFactory = GPUImageFactoryType::New();
        itk::ObjectFactoryBase::RegisterFactory(imageFactory);

        // Create GPU Advanced BSpline transform in explicit mode
        using GPUBSplineTransformType = GPUAdvancedBSplineDeformableTransform<GPUScalarType, SpaceDimension, 0>;
        bsplineTransform = GPUBSplineTransformType::New();

        // UnRegister image factory
        itk::ObjectFactoryBase::UnRegisterFactory(imageFactory);
      }
      else
      {
        // Create GPU Advanced BSpline transform in implicit mode
        using GPUBSplineTransformType = AdvancedBSplineDeformableTransform<GPUScalarType, SpaceDimension, 0>;
        bsplineTransform = GPUBSplineTransformType::New();
      }
      this->CastCopyTransformParameters(fromTransform, bsplineTransform);
      toTransform->SetCurrentTransform(bsplineTransform);
      return true;
    }
    else
    {
      // Try BSpline Order 1
      const typename AdvancedBSplineOrder1TransformType::ConstPointer bsplineOrder1 =
        dynamic_cast<const AdvancedBSplineOrder1TransformType *>(fromTransform.GetPointer());

      if (bsplineOrder1)
      {
        if (this->m_ExplicitMode)
        {
          // Register image factory
          GPUImageFactoryPointer imageFactory = GPUImageFactoryType::New();
          itk::ObjectFactoryBase::RegisterFactory(imageFactory);

          // Create GPU Advanced BSpline transform in explicit mode
          using GPUBSplineTransformType = GPUAdvancedBSplineDeformableTransform<GPUScalarType, SpaceDimension, 1>;
          bsplineTransform = GPUBSplineTransformType::New();

          // UnRegister image factory
          itk::ObjectFactoryBase::UnRegisterFactory(imageFactory);
        }
        else
        {
          // Create GPU Advanced BSpline transform in implicit mode
          using GPUBSplineTransformType = AdvancedBSplineDeformableTransform<GPUScalarType, SpaceDimension, 1>;
          bsplineTransform = GPUBSplineTransformType::New();
        }
        this->CastCopyTransformParameters(fromTransform, bsplineTransform);
        toTransform->SetCurrentTransform(bsplineTransform);
        return true;
      }
      else
      {
        // Try BSpline Order 2
        const typename AdvancedBSplineOrder2TransformType::ConstPointer bsplineOrder2 =
          dynamic_cast<const AdvancedBSplineOrder2TransformType *>(fromTransform.GetPointer());

        if (bsplineOrder2)
        {
          if (this->m_ExplicitMode)
          {
            // Register image factory
            GPUImageFactoryPointer imageFactory = GPUImageFactoryType::New();
            itk::ObjectFactoryBase::RegisterFactory(imageFactory);

            // Create GPU Advanced BSpline transform in explicit mode
            using GPUBSplineTransformType = GPUAdvancedBSplineDeformableTransform<GPUScalarType, SpaceDimension, 2>;
            bsplineTransform = GPUBSplineTransformType::New();

            // UnRegister image factory
            itk::ObjectFactoryBase::UnRegisterFactory(imageFactory);
          }
          else
          {
            // Create GPU Advanced BSpline transform in implicit mode
            using GPUBSplineTransformType = AdvancedBSplineDeformableTransform<GPUScalarType, SpaceDimension, 2>;
            bsplineTransform = GPUBSplineTransformType::New();
          }
          this->CastCopyTransformParameters(fromTransform, bsplineTransform);
          toTransform->SetCurrentTransform(bsplineTransform);
          return true;
        }
      }
    }
  }

  return false;
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
bool
GPUAdvancedCombinationTransformCopier<
  TTypeList,
  NDimensions,
  TAdvancedCombinationTransform,
  TOutputTransformPrecisionType>::CopyEuler2DTransform(const CPUCurrentTransformConstPointer & fromTransform,
                                                       GPUComboTransformPointer &              toTransform,
                                                       TransformSpaceDimensionToType<2>)
{
  using CPUEulerTransformType = AdvancedRigid2DTransform<CPUScalarType>;
  const typename CPUEulerTransformType::ConstPointer euler =
    dynamic_cast<const CPUEulerTransformType *>(fromTransform.GetPointer());

  if (euler)
  {
    GPUAdvancedTransformPointer eulerTransform;
    if (this->m_ExplicitMode)
    {
      // Create GPU Advanced Euler transform in explicit mode
      using GPUEulerTransformType = GPUAdvancedEuler2DTransform<GPUScalarType>;
      eulerTransform = GPUEulerTransformType::New();
    }
    else
    {
      // Create GPU Advanced Euler transform in implicit mode
      using GPUEulerTransformType = AdvancedRigid2DTransform<GPUScalarType>;
      eulerTransform = GPUEulerTransformType::New();
    }

    this->CastCopyTransformParameters(fromTransform, eulerTransform);
    toTransform->SetCurrentTransform(eulerTransform);
    return true;
  }

  return false;
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
bool
GPUAdvancedCombinationTransformCopier<
  TTypeList,
  NDimensions,
  TAdvancedCombinationTransform,
  TOutputTransformPrecisionType>::CopyEuler3DTransform(const CPUCurrentTransformConstPointer & fromTransform,
                                                       GPUComboTransformPointer &              toTransform,
                                                       TransformSpaceDimensionToType<3>)
{
  using CPUEulerTransformType = AdvancedEuler3DTransform<CPUScalarType>;
  const typename CPUEulerTransformType::ConstPointer euler =
    dynamic_cast<const CPUEulerTransformType *>(fromTransform.GetPointer());

  if (euler)
  {
    GPUAdvancedTransformPointer eulerTransform;
    if (this->m_ExplicitMode)
    {
      // Create GPU Advanced Euler transform in explicit mode
      using GPUEulerTransformType = GPUAdvancedEuler3DTransform<GPUScalarType>;
      eulerTransform = GPUEulerTransformType::New();
    }
    else
    {
      // Create GPU Advanced Euler transform in implicit mode
      using GPUEulerTransformType = AdvancedEuler3DTransform<GPUScalarType>;
      eulerTransform = GPUEulerTransformType::New();
    }

    this->CastCopyTransformParameters(fromTransform, eulerTransform);
    toTransform->SetCurrentTransform(eulerTransform);
    return true;
  }

  return false;
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
bool
GPUAdvancedCombinationTransformCopier<
  TTypeList,
  NDimensions,
  TAdvancedCombinationTransform,
  TOutputTransformPrecisionType>::CopySimilarity2DTransform(const CPUCurrentTransformConstPointer & fromTransform,
                                                            GPUComboTransformPointer &              toTransform,
                                                            TransformSpaceDimensionToType<2>)
{
  using CPUSimilarityTransformType = AdvancedSimilarity2DTransform<CPUScalarType>;
  const typename CPUSimilarityTransformType::ConstPointer similarity =
    dynamic_cast<const CPUSimilarityTransformType *>(fromTransform.GetPointer());

  if (similarity)
  {
    GPUAdvancedTransformPointer similarityTransform;
    if (this->m_ExplicitMode)
    {
      // Create GPU Advanced Similarity transform in explicit mode
      using GPUSimilarityTransformType = GPUAdvancedSimilarity2DTransform<GPUScalarType>;
      similarityTransform = GPUSimilarityTransformType::New();
    }
    else
    {
      // Create GPU Advanced Similarity transform in implicit mode
      using GPUSimilarityTransformType = AdvancedSimilarity2DTransform<GPUScalarType>;
      similarityTransform = GPUSimilarityTransformType::New();
    }
    this->CastCopyTransformParameters(fromTransform, similarityTransform);
    toTransform->SetCurrentTransform(similarityTransform);
    return true;
  }

  return false;
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
bool
GPUAdvancedCombinationTransformCopier<
  TTypeList,
  NDimensions,
  TAdvancedCombinationTransform,
  TOutputTransformPrecisionType>::CopySimilarity3DTransform(const CPUCurrentTransformConstPointer & fromTransform,
                                                            GPUComboTransformPointer &              toTransform,
                                                            TransformSpaceDimensionToType<3>)
{
  using CPUSimilarityTransformType = AdvancedSimilarity3DTransform<CPUScalarType>;
  const typename CPUSimilarityTransformType::ConstPointer similarity =
    dynamic_cast<const CPUSimilarityTransformType *>(fromTransform.GetPointer());

  if (similarity)
  {
    GPUAdvancedTransformPointer similarityTransform;
    if (this->m_ExplicitMode)
    {
      // Create GPU Advanced Similarity transform in explicit mode
      using GPUSimilarityTransformType = GPUAdvancedSimilarity3DTransform<GPUScalarType>;
      similarityTransform = GPUSimilarityTransformType::New();
    }
    else
    {
      // Create GPU Advanced Similarity transform in implicit mode
      using GPUSimilarityTransformType = AdvancedSimilarity3DTransform<GPUScalarType>;
      similarityTransform = GPUSimilarityTransformType::New();
    }
    this->CastCopyTransformParameters(fromTransform, similarityTransform);
    toTransform->SetCurrentTransform(similarityTransform);
    return true;
  }

  return false;
}


//------------------------------------------------------------------------------
template <typename TTypeList,
          typename NDimensions,
          typename TAdvancedCombinationTransform,
          typename TOutputTransformPrecisionType>
void
GPUAdvancedCombinationTransformCopier<TTypeList,
                                      NDimensions,
                                      TAdvancedCombinationTransform,
                                      TOutputTransformPrecisionType>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Input Transform: " << this->m_InputTransform << std::endl;
  os << indent << "Output Transform: " << this->m_Output << std::endl;
  os << indent << "Internal Transform Time: " << this->m_InternalTransformTime << std::endl;
  os << indent << "Explicit Mode: " << this->m_ExplicitMode << std::endl;
}


} // end namespace itk

#endif
