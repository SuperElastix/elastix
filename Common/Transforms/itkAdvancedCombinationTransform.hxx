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
#ifndef itkAdvancedCombinationTransform_hxx
#define itkAdvancedCombinationTransform_hxx

#include "itkAdvancedCombinationTransform.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template <typename TScalarType, unsigned int NDimensions>
AdvancedCombinationTransform<TScalarType, NDimensions>::AdvancedCombinationTransform()
  : Superclass(NDimensions)
{}


/**
 *
 * ***********************************************************
 * ***** Override functions to aid for combining transformations.
 *
 * ***********************************************************
 *
 */

/**
 * ***************** GetNumberOfParameters **************************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::GetNumberOfParameters() const -> NumberOfParametersType
{
  /** Return the number of parameters that completely define
   * the m_CurrentTransform.
   */
  if (this->m_CurrentTransform.IsNotNull())
  {
    return this->m_CurrentTransform->GetNumberOfParameters();
  }
  else
  {
    /** Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }

} // end GetNumberOfParameters()


/**
 * ***************** GetNumberOfTransforms **************************
 */

template <typename TScalarType, unsigned int NDimensions>
SizeValueType
AdvancedCombinationTransform<TScalarType, NDimensions>::GetNumberOfTransforms() const
{
  SizeValueType                num = 0;
  CurrentTransformConstPointer currentTransform = GetCurrentTransform();

  if (currentTransform.IsNotNull())
  {
    InitialTransformConstPointer initialTransform = GetInitialTransform();
    if (initialTransform.IsNotNull())
    {
      const Self * initialTransformCasted = dynamic_cast<const Self *>(initialTransform.GetPointer());
      if (initialTransformCasted)
      {
        num += initialTransformCasted->GetNumberOfTransforms() + 1;
      }
    }
    else
    {
      ++num;
    }
  }

  return num;
} // end GetNumberOfTransforms()


/**
 * ***************** GetNthTransform **************************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::GetNthTransform(SizeValueType n) const
  -> const TransformTypePointer
{
  const SizeValueType numTransforms = GetNumberOfTransforms();
  if (n > numTransforms - 1)
  {
    itkExceptionMacro(<< "The AdvancedCombinationTransform contains " << numTransforms
                      << " transforms. Unable to retrieve Nth current transform with index " << n);
  }

  TransformTypePointer               nthTransform;
  const CurrentTransformConstPointer currentTransform = GetCurrentTransform();

  if (currentTransform.IsNotNull())
  {
    if (n == 0)
    {
      // Perform const_cast, we don't like it, but there is no other option
      // with current ITK4 itk::MultiTransform::GetNthTransform() const design
      const TransformType * currentTransformCasted = dynamic_cast<const TransformType *>(currentTransform.GetPointer());
      TransformType *       currentTransformConstCasted = const_cast<TransformType *>(currentTransformCasted);
      nthTransform = currentTransformConstCasted;
    }
    else
    {
      const InitialTransformConstPointer initialTransform = GetInitialTransform();
      if (initialTransform.IsNotNull())
      {
        const Self * initialTransformCasted = dynamic_cast<const Self *>(initialTransform.GetPointer());
        if (initialTransformCasted)
        {
          const SizeValueType id = n - 1;
          nthTransform = initialTransformCasted->GetNthTransform(id);
        }
      }
    }
  }

  return nthTransform;
} // end GetNthTransform()


/**
 * ***************** GetNumberOfNonZeroJacobianIndices **************************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::GetNumberOfNonZeroJacobianIndices() const
  -> NumberOfParametersType
{
  /** Return the number of parameters that completely define
   * the m_CurrentTransform.
   */
  if (this->m_CurrentTransform.IsNotNull())
  {
    return this->m_CurrentTransform->GetNumberOfNonZeroJacobianIndices();
  }
  else
  {
    /** Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }

} // end GetNumberOfNonZeroJacobianIndices()


/**
 * ***************** IsLinear **************************
 */

template <typename TScalarType, unsigned int NDimensions>
bool
AdvancedCombinationTransform<TScalarType, NDimensions>::IsLinear() const
{
  const auto isTransformLinear = [](const auto & transform) { return transform.IsNull() || transform->IsLinear(); };

  return isTransformLinear(m_CurrentTransform) && isTransformLinear(m_InitialTransform);

} // end IsLinear()


/**
 * ***************** GetTransformCategory **************************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::GetTransformCategory() const -> TransformCategoryEnum
{
  // Check if all linear
  if (this->IsLinear())
  {
    return TransformCategoryEnum::Linear;
  }

  // It is unclear how you would prefer to define the rest of them,
  // lets just return Self::UnknownTransformCategory for now
  return TransformCategoryEnum::UnknownTransformCategory;
}


/**
 * ***************** GetParameters **************************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::GetParameters() const -> const ParametersType &
{
  /** Return the parameters that completely define the m_CurrentTransform. */
  if (this->m_CurrentTransform.IsNotNull())
  {
    return this->m_CurrentTransform->GetParameters();
  }
  else
  {
    /** Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }

} // end GetParameters()

/**
 * ***************** GetFixedParameters **************************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::GetFixedParameters() const -> const FixedParametersType &
{
  /** Return the fixed parameters that define the m_CurrentTransform. */
  if (this->m_CurrentTransform.IsNotNull())
  {
    return this->m_CurrentTransform->GetFixedParameters();
  }
  else
  {
    /** Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }

} // end GetFixedParameters()

/**
 * ***************** SetParameters **************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::SetParameters(const ParametersType & param)
{
  /** Set the parameters in the m_CurrentTransform. */
  if (this->m_CurrentTransform.IsNotNull())
  {
    this->Modified();
    this->m_CurrentTransform->SetParameters(param);
  }
  else
  {
    /** Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }

} // end SetParameters()


/**
 * ***************** SetFixedParameters **************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::SetFixedParameters(const FixedParametersType & param)
{
  /** Set the parameters in the m_CurrentTransform. */
  if (this->m_CurrentTransform.IsNotNull())
  {
    this->Modified();
    this->m_CurrentTransform->SetFixedParameters(param);
  }
  else
  {
    /** Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }

} // end SetFixedParameters()


/**
 * ***************** SetParametersByValue **************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::SetParametersByValue(const ParametersType & param)
{
  /** Set the parameters in the m_CurrentTransfom. */
  if (this->m_CurrentTransform.IsNotNull())
  {
    this->Modified();
    this->m_CurrentTransform->SetParametersByValue(param);
  }
  else
  {
    /** Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }

} // end SetParametersByValue()


/**
 * ***************** GetInverse **************************
 */

template <typename TScalarType, unsigned int NDimensions>
bool
AdvancedCombinationTransform<TScalarType, NDimensions>::GetInverse(Self * inverse) const
{
  if (!inverse)
  {
    /** Inverse transformation cannot be returned into nothingness. */
    return false;
  }
  else if (this->m_CurrentTransform.IsNull())
  {
    /** No current transform has been set. Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }
  else if (this->m_InitialTransform.IsNull())
  {
    /** No Initial transform, so call the CurrentTransform's implementation. */
    return this->m_CurrentTransform->GetInverse(inverse);
  }
  else if (this->m_UseAddition)
  {
    /** No generic expression exists for the inverse of (T0+T1)(x). */
    return false;
  }
  else // UseComposition
  {
    /** The initial transform and the current transform have been set
     * and UseComposition is set to true.
     * The inverse transform IT is defined by:
     *  IT ( T1(T0(x) ) = x
     * So:
     *  IT(y) = T0^{-1} ( T1^{-1} (y) ),
     * which is of course only defined when the inverses of both
     * the initial and the current transforms are defined.
     */

    itkExceptionMacro(<< "ERROR: not implemented");

    //     /** Try create the inverse of the initial transform. */
    //     InitialTransformPointer inverseT0 = InitialTransformType::New();
    //     bool T0invertable = this->m_InitialTransform->GetInverse( inverseT0 );
    //
    //     if ( T0invertable )
    //     {
    //       /** Try to create the inverse of the current transform. */
    //       CurrentTransformPointer inverseT1 = CurrentTransformType::New();
    //       bool T1invertable = this->m_CurrentTransform->GetInverse( inverseT1 );
    //
    //       if ( T1invertable )
    //       {
    //         /** The transform can be inverted! */
    //         inverse->SetUseComposition( true );
    //         inverse->SetInitialTransform( inverseT1 );
    //         inverse->SetCurrentTransform( inverseT0 );
    //         return true;
    //       }
    //       else
    //       {
    //         /** The initial transform is invertible, but the current one not. */
    //         return false;
    //       }
    //     }
    //     else
    //     {
    //       /** The initial transform is not invertible. */
    //       return false;
    //     }
  } // end else: UseComposition

} // end GetInverse()


/**
 * ***************** GetHasNonZeroSpatialHessian **************************
 */

template <typename TScalarType, unsigned int NDimensions>
bool
AdvancedCombinationTransform<TScalarType, NDimensions>::GetHasNonZeroSpatialHessian() const
{
  /** Set the parameters in the m_CurrentTransfom. */
  if (this->m_CurrentTransform.IsNull())
  {
    /** No current transform has been set. Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }
  else if (this->m_InitialTransform.IsNull())
  {
    /** No Initial transform, so call the CurrentTransform's implementation. */
    return this->m_CurrentTransform->GetHasNonZeroSpatialHessian();
  }
  else
  {
    bool dummy = this->m_InitialTransform->GetHasNonZeroSpatialHessian() ||
                 this->m_CurrentTransform->GetHasNonZeroSpatialHessian();
    return dummy;
  }

} // end GetHasNonZeroSpatialHessian()


/**
 * ***************** HasNonZeroJacobianOfSpatialHessian **************************
 */

template <typename TScalarType, unsigned int NDimensions>
bool
AdvancedCombinationTransform<TScalarType, NDimensions>::HasNonZeroJacobianOfSpatialHessian() const
{
  /** Set the parameters in the m_CurrentTransfom. */
  if (this->m_CurrentTransform.IsNull())
  {
    /** No current transform has been set. Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }
  else if (this->m_InitialTransform.IsNull())
  {
    /** No Initial transform, so call the CurrentTransform's implementation. */
    return this->m_CurrentTransform->GetHasNonZeroJacobianOfSpatialHessian();
  }
  else
  {
    bool dummy = this->m_InitialTransform->GetHasNonZeroJacobianOfSpatialHessian() ||
                 this->m_CurrentTransform->GetHasNonZeroJacobianOfSpatialHessian();
    return dummy;
  }

} // end HasNonZeroJacobianOfSpatialHessian()


/**
 *
 * ***********************************************************
 * ***** Functions to set the transformations and choose the
 * ***** combination method.
 *
 * ***********************************************************
 *
 */

/**
 * ******************* SetInitialTransform **********************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::SetInitialTransform(InitialTransformType * _arg)
{
  /** Set the the initial transform and call the UpdateCombinationMethod. */
  if (this->m_InitialTransform != _arg)
  {
    this->m_InitialTransform = _arg;
    this->Modified();
    this->UpdateCombinationMethod();
  }

} // end SetInitialTransform()


/**
 * ******************* SetCurrentTransform **********************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::SetCurrentTransform(CurrentTransformType * _arg)
{
  /** Set the the current transform and call the UpdateCombinationMethod. */
  if (this->m_CurrentTransform != _arg)
  {
    this->m_CurrentTransform = _arg;
    this->Modified();
    this->UpdateCombinationMethod();
  }

} // end SetCurrentTransform()


/**
 * ********************** SetUseAddition **********************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::SetUseAddition(bool _arg)
{
  /** Set the UseAddition and UseComposition bools and call the UpdateCombinationMethod. */
  if (this->m_UseAddition != _arg)
  {
    this->m_UseAddition = _arg;
    this->m_UseComposition = !_arg;
    this->Modified();
    this->UpdateCombinationMethod();
  }

} // end SetUseAddition()


/**
 * ********************** SetUseComposition *******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::SetUseComposition(bool _arg)
{
  /** Set the UseAddition and UseComposition bools and call the UpdateCombinationMethod. */
  if (this->m_UseComposition != _arg)
  {
    this->m_UseComposition = _arg;
    this->m_UseAddition = !_arg;
    this->Modified();
    this->UpdateCombinationMethod();
  }

} // end SetUseComposition()


/**
 * ****************** UpdateCombinationMethod ********************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::UpdateCombinationMethod()
{
  /** Update the m_SelectedTransformPointFunction and
   * the m_SelectedGetJacobianFunction
   */
  if (this->m_CurrentTransform.IsNull())
  {
    this->m_SelectedTransformPointFunction = &Self::TransformPointNoCurrentTransform;
    //     this->m_SelectedGetJacobianFunction
    //       = &Self::GetJacobianNoCurrentTransform;
    this->m_SelectedGetSparseJacobianFunction = &Self::GetJacobianNoCurrentTransform;
    this->m_SelectedEvaluateJacobianWithImageGradientProductFunction =
      &Self::EvaluateJacobianWithImageGradientProductNoCurrentTransform;
    this->m_SelectedGetSpatialJacobianFunction = &Self::GetSpatialJacobianNoCurrentTransform;
    this->m_SelectedGetSpatialHessianFunction = &Self::GetSpatialHessianNoCurrentTransform;
    this->m_SelectedGetJacobianOfSpatialJacobianFunction = &Self::GetJacobianOfSpatialJacobianNoCurrentTransform;
    this->m_SelectedGetJacobianOfSpatialJacobianFunction2 = &Self::GetJacobianOfSpatialJacobianNoCurrentTransform;
    this->m_SelectedGetJacobianOfSpatialHessianFunction = &Self::GetJacobianOfSpatialHessianNoCurrentTransform;
    this->m_SelectedGetJacobianOfSpatialHessianFunction2 = &Self::GetJacobianOfSpatialHessianNoCurrentTransform;
  }
  else if (this->m_InitialTransform.IsNull())
  {
    this->m_SelectedTransformPointFunction = &Self::TransformPointNoInitialTransform;
    //     this->m_SelectedGetJacobianFunction
    //       = &Self::GetJacobianNoInitialTransform;
    this->m_SelectedGetSparseJacobianFunction = &Self::GetJacobianNoInitialTransform;
    this->m_SelectedEvaluateJacobianWithImageGradientProductFunction =
      &Self::EvaluateJacobianWithImageGradientProductNoInitialTransform;
    this->m_SelectedGetSpatialJacobianFunction = &Self::GetSpatialJacobianNoInitialTransform;
    this->m_SelectedGetSpatialHessianFunction = &Self::GetSpatialHessianNoInitialTransform;
    this->m_SelectedGetJacobianOfSpatialJacobianFunction = &Self::GetJacobianOfSpatialJacobianNoInitialTransform;
    this->m_SelectedGetJacobianOfSpatialJacobianFunction2 = &Self::GetJacobianOfSpatialJacobianNoInitialTransform;
    this->m_SelectedGetJacobianOfSpatialHessianFunction = &Self::GetJacobianOfSpatialHessianNoInitialTransform;
    this->m_SelectedGetJacobianOfSpatialHessianFunction2 = &Self::GetJacobianOfSpatialHessianNoInitialTransform;
  }
  else if (this->m_UseAddition)
  {
    this->m_SelectedTransformPointFunction = &Self::TransformPointUseAddition;
    //     this->m_SelectedGetJacobianFunction
    //       = &Self::GetJacobianUseAddition;
    this->m_SelectedGetSparseJacobianFunction = &Self::GetJacobianUseAddition;
    this->m_SelectedEvaluateJacobianWithImageGradientProductFunction =
      &Self::EvaluateJacobianWithImageGradientProductUseAddition;
    this->m_SelectedGetSpatialJacobianFunction = &Self::GetSpatialJacobianUseAddition;
    this->m_SelectedGetSpatialHessianFunction = &Self::GetSpatialHessianUseAddition;
    this->m_SelectedGetJacobianOfSpatialJacobianFunction = &Self::GetJacobianOfSpatialJacobianUseAddition;
    this->m_SelectedGetJacobianOfSpatialJacobianFunction2 = &Self::GetJacobianOfSpatialJacobianUseAddition;
    this->m_SelectedGetJacobianOfSpatialHessianFunction = &Self::GetJacobianOfSpatialHessianUseAddition;
    this->m_SelectedGetJacobianOfSpatialHessianFunction2 = &Self::GetJacobianOfSpatialHessianUseAddition;
  }
  else
  {
    this->m_SelectedTransformPointFunction = &Self::TransformPointUseComposition;
    //     this->m_SelectedGetJacobianFunction
    //       = &Self::GetJacobianUseComposition;
    this->m_SelectedGetSparseJacobianFunction = &Self::GetJacobianUseComposition;
    this->m_SelectedEvaluateJacobianWithImageGradientProductFunction =
      &Self::EvaluateJacobianWithImageGradientProductUseComposition;
    this->m_SelectedGetSpatialJacobianFunction = &Self::GetSpatialJacobianUseComposition;
    this->m_SelectedGetSpatialHessianFunction = &Self::GetSpatialHessianUseComposition;
    this->m_SelectedGetJacobianOfSpatialJacobianFunction = &Self::GetJacobianOfSpatialJacobianUseComposition;
    this->m_SelectedGetJacobianOfSpatialJacobianFunction2 = &Self::GetJacobianOfSpatialJacobianUseComposition;
    this->m_SelectedGetJacobianOfSpatialHessianFunction = &Self::GetJacobianOfSpatialHessianUseComposition;
    this->m_SelectedGetJacobianOfSpatialHessianFunction2 = &Self::GetJacobianOfSpatialHessianUseComposition;
  }

} // end UpdateCombinationMethod()


/**
 *
 * ***********************************************************
 * ***** Functions that implement the:
 * ***** - TransformPoint()
 * ***** - GetJacobian()
 * ***** - EvaluateJacobianWithImageGradientProduct()
 * ***** - GetSpatialJacobian()
 * ***** - GetSpatialHessian()
 * ***** - GetJacobianOfSpatialJacobian()
 * ***** - GetJacobianOfSpatialHessian()
 * ***** for the four possible cases:
 * ***** - no initial transform: this is the same as using only one transform
 * ***** - no current transform: error, it should be set
 * ***** - use addition to combine transformations
 * ***** - use composition to combine transformations
 *
 * ***********************************************************
 *
 */

/**
 * ************* TransformPointUseAddition **********************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::TransformPointUseAddition(const InputPointType & point) const
  -> OutputPointType
{
  /** The Initial transform. */
  OutputPointType out0 = this->m_InitialTransform->TransformPoint(point);

  /** The Current transform. */
  OutputPointType out = this->m_CurrentTransform->TransformPoint(point);

  /** Add them. */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    out[i] += (out0[i] - point[i]);
  }

  return out;

} // end TransformPointUseAddition()


/**
 * **************** TransformPointUseComposition *************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::TransformPointUseComposition(const InputPointType & point) const
  -> OutputPointType
{
  return this->m_CurrentTransform->TransformPoint(this->m_InitialTransform->TransformPoint(point));

} // end TransformPointUseComposition()


/**
 * **************** TransformPointNoInitialTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::TransformPointNoInitialTransform(
  const InputPointType & point) const -> OutputPointType
{
  return this->m_CurrentTransform->TransformPoint(point);

} // end TransformPointNoInitialTransform()


/**
 * ******** TransformPointNoCurrentTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::TransformPointNoCurrentTransform(
  const InputPointType & point) const -> OutputPointType
{
  /** Throw an exception. */
  itkExceptionMacro(<< NoCurrentTransformSet);

} // end TransformPointNoCurrentTransform()


/**
 * ************* GetJacobianUseAddition ***************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianUseAddition(
  const InputPointType &       inputPoint,
  JacobianType &               j,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->GetJacobian(inputPoint, j, nonZeroJacobianIndices);

} // end GetJacobianUseAddition()


/**
 * **************** GetJacobianUseComposition *************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianUseComposition(
  const InputPointType &       inputPoint,
  JacobianType &               j,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->GetJacobian(
    this->m_InitialTransform->TransformPoint(inputPoint), j, nonZeroJacobianIndices);

} // end GetJacobianUseComposition()


/**
 * **************** GetJacobianNoInitialTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianNoInitialTransform(
  const InputPointType &       inputPoint,
  JacobianType &               j,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->GetJacobian(inputPoint, j, nonZeroJacobianIndices);

} // end GetJacobianNoInitialTransform()


/**
 * ******** GetJacobianNoCurrentTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianNoCurrentTransform(
  const InputPointType &       itkNotUsed(inputPoint),
  JacobianType &               itkNotUsed(j),
  NonZeroJacobianIndicesType & itkNotUsed(nonZeroJacobianIndices)) const
{
  /** Throw an exception. */
  itkExceptionMacro(<< NoCurrentTransformSet);

} // end GetJacobianNoCurrentTransform()


/**
 * ************* EvaluateJacobianWithImageGradientProductUseAddition ***************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::EvaluateJacobianWithImageGradientProductUseAddition(
  const InputPointType &          inputPoint,
  const MovingImageGradientType & movingImageGradient,
  DerivativeType &                imageJacobian,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->EvaluateJacobianWithImageGradientProduct(
    inputPoint, movingImageGradient, imageJacobian, nonZeroJacobianIndices);

} // end EvaluateJacobianWithImageGradientProductUseAddition()


/**
 * **************** EvaluateJacobianWithImageGradientProductUseComposition *************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::EvaluateJacobianWithImageGradientProductUseComposition(
  const InputPointType &          inputPoint,
  const MovingImageGradientType & movingImageGradient,
  DerivativeType &                imageJacobian,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->EvaluateJacobianWithImageGradientProduct(
    this->m_InitialTransform->TransformPoint(inputPoint), movingImageGradient, imageJacobian, nonZeroJacobianIndices);

} // end EvaluateJacobianWithImageGradientProductUseComposition()


/**
 * **************** EvaluateJacobianWithImageGradientProductNoInitialTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::EvaluateJacobianWithImageGradientProductNoInitialTransform(
  const InputPointType &          inputPoint,
  const MovingImageGradientType & movingImageGradient,
  DerivativeType &                imageJacobian,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->EvaluateJacobianWithImageGradientProduct(
    inputPoint, movingImageGradient, imageJacobian, nonZeroJacobianIndices);

} // end EvaluateJacobianWithImageGradientProductNoInitialTransform()


/**
 * ******** EvaluateJacobianWithImageGradientProductNoCurrentTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::EvaluateJacobianWithImageGradientProductNoCurrentTransform(
  const InputPointType &          inputPoint,
  const MovingImageGradientType & movingImageGradient,
  DerivativeType &                imageJacobian,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  /** Throw an exception. */
  itkExceptionMacro(<< NoCurrentTransformSet);

} // end EvaluateJacobianWithImageGradientProductNoCurrentTransform()


/**
 * ************* GetSpatialJacobianUseAddition ***************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialJacobianUseAddition(const InputPointType & inputPoint,
                                                                                      SpatialJacobianType &  sj) const
{
  SpatialJacobianType sj0, sj1;
  this->m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
  this->m_CurrentTransform->GetSpatialJacobian(inputPoint, sj1);
  sj = sj0 + sj1 - SpatialJacobianType::GetIdentity();

} // end GetSpatialJacobianUseAddition()


/**
 * **************** GetSpatialJacobianUseComposition *************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialJacobianUseComposition(
  const InputPointType & inputPoint,
  SpatialJacobianType &  sj) const
{
  SpatialJacobianType sj0, sj1;
  this->m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
  this->m_CurrentTransform->GetSpatialJacobian(this->m_InitialTransform->TransformPoint(inputPoint), sj1);

  sj = sj1 * sj0;

} // end GetSpatialJacobianUseComposition()


/**
 * **************** GetSpatialJacobianNoInitialTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialJacobianNoInitialTransform(
  const InputPointType & inputPoint,
  SpatialJacobianType &  sj) const
{
  this->m_CurrentTransform->GetSpatialJacobian(inputPoint, sj);

} // end GetSpatialJacobianNoInitialTransform()


/**
 * ******** GetSpatialJacobianNoCurrentTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialJacobianNoCurrentTransform(
  const InputPointType & itkNotUsed(inputPoint),
  SpatialJacobianType &  itkNotUsed(sj)) const
{
  /** Throw an exception. */
  itkExceptionMacro(<< NoCurrentTransformSet);

} // end GetSpatialJacobianNoCurrentTransform()


/**
 * ******** GetSpatialHessianUseAddition ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialHessianUseAddition(const InputPointType & inputPoint,
                                                                                     SpatialHessianType &   sh) const
{
  SpatialHessianType sh0, sh1;
  this->m_InitialTransform->GetSpatialHessian(inputPoint, sh0);
  this->m_CurrentTransform->GetSpatialHessian(inputPoint, sh1);

  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    sh[i] = sh0[i] + sh1[i];
  }

} // end GetSpatialHessianUseAddition()


/**
 * ******** GetSpatialHessianUseComposition ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialHessianUseComposition(
  const InputPointType & inputPoint,
  SpatialHessianType &   sh) const
{
  /** Create intermediary variables for the internal transforms. */
  SpatialJacobianType sj0, sj1;
  SpatialHessianType  sh0, sh1;

  /** Transform the input point. */
  // \todo this has already been computed and it is expensive.
  InputPointType transformedPoint = this->m_InitialTransform->TransformPoint(inputPoint);

  /** Compute the (Jacobian of the) spatial Jacobian / Hessian of the
   * internal transforms.
   */
  this->m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
  this->m_CurrentTransform->GetSpatialJacobian(transformedPoint, sj1);
  this->m_InitialTransform->GetSpatialHessian(inputPoint, sh0);
  this->m_CurrentTransform->GetSpatialHessian(transformedPoint, sh1);

  typename SpatialJacobianType::InternalMatrixType sj0tvnl = sj0.GetTranspose();
  SpatialJacobianType                              sj0t(sj0tvnl);

  /** Combine them in one overall spatial Hessian. */
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    sh[dim] = sj0t * (sh1[dim] * sj0);

    for (unsigned int p = 0; p < SpaceDimension; ++p)
    {
      sh[dim] += (sh0[p] * sj1(dim, p));
    }
  }

} // end GetSpatialHessianUseComposition()


/**
 * ******** GetSpatialHessianNoInitialTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialHessianNoInitialTransform(
  const InputPointType & inputPoint,
  SpatialHessianType &   sh) const
{
  this->m_CurrentTransform->GetSpatialHessian(inputPoint, sh);

} // end GetSpatialHessianNoInitialTransform()


/**
 * ******** GetSpatialHessianNoCurrentTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialHessianNoCurrentTransform(
  const InputPointType & itkNotUsed(inputPoint),
  SpatialHessianType &   itkNotUsed(sh)) const
{
  /** Throw an exception. */
  itkExceptionMacro(<< NoCurrentTransformSet);

} // end GetSpatialHessianNoCurrentTransform()


/**
 * ******** GetJacobianOfSpatialJacobianUseAddition ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialJacobianUseAddition(
  const InputPointType &          inputPoint,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->GetJacobianOfSpatialJacobian(inputPoint, jsj, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialJacobianUseAddition()


/**
 * ******** GetJacobianOfSpatialJacobianUseAddition ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialJacobianUseAddition(
  const InputPointType &          inputPoint,
  SpatialJacobianType &           sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->GetJacobianOfSpatialJacobian(inputPoint, sj, jsj, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialJacobianUseAddition()


/**
 * ******** GetJacobianOfSpatialJacobianUseComposition ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialJacobianUseComposition(
  const InputPointType &          inputPoint,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  SpatialJacobianType           sj0;
  JacobianOfSpatialJacobianType jsj1;
  this->m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
  this->m_CurrentTransform->GetJacobianOfSpatialJacobian(
    this->m_InitialTransform->TransformPoint(inputPoint), jsj1, nonZeroJacobianIndices);

  jsj.resize(nonZeroJacobianIndices.size());
  for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
  {
    jsj[mu] = jsj1[mu] * sj0;
  }

} // end GetJacobianOfSpatialJacobianUseComposition()


/**
 * ******** GetJacobianOfSpatialJacobianUseComposition ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialJacobianUseComposition(
  const InputPointType &          inputPoint,
  SpatialJacobianType &           sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  SpatialJacobianType           sj0, sj1;
  JacobianOfSpatialJacobianType jsj1;
  this->m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
  this->m_CurrentTransform->GetJacobianOfSpatialJacobian(
    this->m_InitialTransform->TransformPoint(inputPoint), sj1, jsj1, nonZeroJacobianIndices);

  sj = sj1 * sj0;
  jsj.resize(nonZeroJacobianIndices.size());
  for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
  {
    jsj[mu] = jsj1[mu] * sj0;
  }

} // end GetJacobianOfSpatialJacobianUseComposition()


/**
 * ******** GetJacobianOfSpatialJacobianNoInitialTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialJacobianNoInitialTransform(
  const InputPointType &          inputPoint,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->GetJacobianOfSpatialJacobian(inputPoint, jsj, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialJacobianNoInitialTransform()


/**
 * ******** GetJacobianOfSpatialJacobianNoInitialTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialJacobianNoInitialTransform(
  const InputPointType &          inputPoint,
  SpatialJacobianType &           sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->GetJacobianOfSpatialJacobian(inputPoint, sj, jsj, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialJacobianNoInitialTransform()


/**
 * ******** GetJacobianOfSpatialJacobianNoCurrentTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialJacobianNoCurrentTransform(
  const InputPointType &          itkNotUsed(inputPoint),
  JacobianOfSpatialJacobianType & itkNotUsed(jsj),
  NonZeroJacobianIndicesType &    itkNotUsed(nonZeroJacobianIndices)) const
{
  /** Throw an exception. */
  itkExceptionMacro(<< NoCurrentTransformSet);

} // end GetJacobianOfSpatialJacobianNoCurrentTransform()


/**
 * ******** GetJacobianOfSpatialJacobianNoCurrentTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialJacobianNoCurrentTransform(
  const InputPointType &          itkNotUsed(inputPoint),
  SpatialJacobianType &           itkNotUsed(sj),
  JacobianOfSpatialJacobianType & itkNotUsed(jsj),
  NonZeroJacobianIndicesType &    itkNotUsed(nonZeroJacobianIndices)) const
{
  /** Throw an exception. */
  itkExceptionMacro(<< NoCurrentTransformSet);

} // end GetJacobianOfSpatialJacobianNoCurrentTransform()


/**
 * ******** GetJacobianOfSpatialHessianUseAddition ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialHessianUseAddition(
  const InputPointType &         inputPoint,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->GetJacobianOfSpatialHessian(inputPoint, jsh, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialHessianUseAddition()


/**
 * ******** GetJacobianOfSpatialHessianUseAddition ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialHessianUseAddition(
  const InputPointType &         inputPoint,
  SpatialHessianType &           sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->GetJacobianOfSpatialHessian(inputPoint, sh, jsh, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialHessianUseAddition()


/**
 * ******** GetJacobianOfSpatialHessianUseComposition ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialHessianUseComposition(
  const InputPointType &         inputPoint,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  /** Create intermediary variables for the internal transforms. */
  SpatialJacobianType           sj0;
  SpatialHessianType            sh0;
  JacobianOfSpatialJacobianType jsj1;
  JacobianOfSpatialHessianType  jsh1;

  /** Transform the input point. */
  // \todo: this has already been computed and it is expensive.
  InputPointType transformedPoint = this->m_InitialTransform->TransformPoint(inputPoint);

  /** Compute the (Jacobian of the) spatial Jacobian / Hessian of the
   * internal transforms. */
  this->m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
  this->m_InitialTransform->GetSpatialHessian(inputPoint, sh0);

  /** Assume/demand that GetJacobianOfSpatialJacobian returns
   * the same nonZeroJacobianIndices as the GetJacobianOfSpatialHessian. */
  this->m_CurrentTransform->GetJacobianOfSpatialJacobian(transformedPoint, jsj1, nonZeroJacobianIndices);
  this->m_CurrentTransform->GetJacobianOfSpatialHessian(transformedPoint, jsh1, nonZeroJacobianIndices);

  typename SpatialJacobianType::InternalMatrixType sj0tvnl = sj0.GetTranspose();
  SpatialJacobianType                              sj0t(sj0tvnl);

  jsh.resize(nonZeroJacobianIndices.size());

  /** Combine them in one overall Jacobian of spatial Hessian. */
  for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
  {
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      jsh[mu][dim] = sj0t * (jsh1[mu][dim] * sj0);
    }
  }

  if (this->m_InitialTransform->GetHasNonZeroSpatialHessian())
  {
    for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
    {
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        for (unsigned int p = 0; p < SpaceDimension; ++p)
        {
          jsh[mu][dim] += (sh0[p] * jsj1[mu](dim, p));
        }
      }
    }
  }

} // end GetJacobianOfSpatialHessianUseComposition()


/**
 * ******** GetJacobianOfSpatialHessianUseComposition ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialHessianUseComposition(
  const InputPointType &         inputPoint,
  SpatialHessianType &           sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  /** Create intermediary variables for the internal transforms. */
  SpatialJacobianType           sj0, sj1;
  SpatialHessianType            sh0, sh1;
  JacobianOfSpatialJacobianType jsj1;
  JacobianOfSpatialHessianType  jsh1;

  /** Transform the input point. */
  // \todo this has already been computed and it is expensive.
  InputPointType transformedPoint = this->m_InitialTransform->TransformPoint(inputPoint);

  /** Compute the (Jacobian of the) spatial Jacobian / Hessian of the
   * internal transforms.
   */
  this->m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
  this->m_InitialTransform->GetSpatialHessian(inputPoint, sh0);

  /** Assume/demand that GetJacobianOfSpatialJacobian returns the same
   * nonZeroJacobianIndices as the GetJacobianOfSpatialHessian.
   */
  this->m_CurrentTransform->GetJacobianOfSpatialJacobian(transformedPoint, sj1, jsj1, nonZeroJacobianIndices);
  this->m_CurrentTransform->GetJacobianOfSpatialHessian(transformedPoint, sh1, jsh1, nonZeroJacobianIndices);

  typename SpatialJacobianType::InternalMatrixType sj0tvnl = sj0.GetTranspose();
  SpatialJacobianType                              sj0t(sj0tvnl);
  jsh.resize(nonZeroJacobianIndices.size());

  /** Combine them in one overall Jacobian of spatial Hessian. */
  for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
  {
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      jsh[mu][dim] = sj0t * (jsh1[mu][dim] * sj0);
    }
  }

  if (this->m_InitialTransform->GetHasNonZeroSpatialHessian())
  {
    for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
    {
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        for (unsigned int p = 0; p < SpaceDimension; ++p)
        {
          jsh[mu][dim] += (sh0[p] * jsj1[mu](dim, p));
        }
      }
    }
  }

  /** Combine them in one overall spatial Hessian. */
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    sh[dim] = sj0t * (sh1[dim] * sj0);
  }

  if (this->m_InitialTransform->GetHasNonZeroSpatialHessian())
  {
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      for (unsigned int p = 0; p < SpaceDimension; ++p)
      {
        sh[dim] += (sh0[p] * sj1(dim, p));
      }
    }
  }

} // end GetJacobianOfSpatialHessianUseComposition()


/**
 * ******** GetJacobianOfSpatialHessianNoInitialTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialHessianNoInitialTransform(
  const InputPointType &         inputPoint,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->GetJacobianOfSpatialHessian(inputPoint, jsh, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialHessianNoInitialTransform()


/**
 * ******** GetJacobianOfSpatialHessianNoInitialTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialHessianNoInitialTransform(
  const InputPointType &         inputPoint,
  SpatialHessianType &           sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  this->m_CurrentTransform->GetJacobianOfSpatialHessian(inputPoint, sh, jsh, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialHessianNoInitialTransform()


/**
 * ******** GetJacobianOfSpatialHessianNoCurrentTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialHessianNoCurrentTransform(
  const InputPointType &         itkNotUsed(inputPoint),
  JacobianOfSpatialHessianType & itkNotUsed(jsh),
  NonZeroJacobianIndicesType &   itkNotUsed(nonZeroJacobianIndices)) const
{
  /** Throw an exception. */
  itkExceptionMacro(<< NoCurrentTransformSet);

} // end GetJacobianOfSpatialHessianNoCurrentTransform()


/**
 * ******** GetJacobianOfSpatialHessianNoCurrentTransform ******************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialHessianNoCurrentTransform(
  const InputPointType &         itkNotUsed(inputPoint),
  SpatialHessianType &           itkNotUsed(sh),
  JacobianOfSpatialHessianType & itkNotUsed(jsh),
  NonZeroJacobianIndicesType &   itkNotUsed(nonZeroJacobianIndices)) const
{
  /** Throw an exception. */
  itkExceptionMacro(<< NoCurrentTransformSet);

} // end GetJacobianOfSpatialHessianNoCurrentTransform()


/**
 *
 * ***********************************************************
 * ***** Functions that point to the selected implementation.
 *
 * ***********************************************************
 *
 */

/**
 * ****************** TransformPoint ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::TransformPoint(const InputPointType & point) const
  -> OutputPointType
{
  /** Call the selected TransformPoint. */
  return (this->*m_SelectedTransformPointFunction)(point);

} // end TransformPoint()


/**
 * ****************** GetJacobian ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobian(
  const InputPointType &       inputPoint,
  JacobianType &               j,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices) const
{
  /** Call the selected GetJacobian. */
  return (this->*m_SelectedGetSparseJacobianFunction)(inputPoint, j, nonZeroJacobianIndices);

} // end GetJacobian()


/**
 * ****************** EvaluateJacobianWithImageGradientProduct ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::EvaluateJacobianWithImageGradientProduct(
  const InputPointType &          inputPoint,
  const MovingImageGradientType & movingImageGradient,
  DerivativeType &                imageJacobian,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  /** Call the selected EvaluateJacobianWithImageGradientProduct. */
  return (this->*m_SelectedEvaluateJacobianWithImageGradientProductFunction)(
    inputPoint, movingImageGradient, imageJacobian, nonZeroJacobianIndices);

} // end EvaluateJacobianWithImageGradientProduct()


/**
 * ****************** GetSpatialJacobian ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialJacobian(const InputPointType & inputPoint,
                                                                           SpatialJacobianType &  sj) const
{
  /** Call the selected GetSpatialJacobian. */
  return (this->*m_SelectedGetSpatialJacobianFunction)(inputPoint, sj);

} // end GetSpatialJacobian()


/**
 * ****************** GetSpatialHessian ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialHessian(const InputPointType & inputPoint,
                                                                          SpatialHessianType &   sh) const
{
  /** Call the selected GetSpatialHessian. */
  return (this->*m_SelectedGetSpatialHessianFunction)(inputPoint, sh);

} // end GetSpatialHessian()


/**
 * ****************** GetJacobianOfSpatialJacobian ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialJacobian(
  const InputPointType &          inputPoint,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  /** Call the selected GetJacobianOfSpatialJacobian. */
  return (this->*m_SelectedGetJacobianOfSpatialJacobianFunction)(inputPoint, jsj, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialJacobian()


/**
 * ****************** GetJacobianOfSpatialJacobian ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialJacobian(
  const InputPointType &          inputPoint,
  SpatialJacobianType &           sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  /** Call the selected GetJacobianOfSpatialJacobian. */
  return (this->*m_SelectedGetJacobianOfSpatialJacobianFunction2)(inputPoint, sj, jsj, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialJacobian()


/**
 * ****************** GetJacobianOfSpatialHessian ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialHessian(
  const InputPointType &         inputPoint,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  /** Call the selected GetJacobianOfSpatialHessian. */
  return (this->*m_SelectedGetJacobianOfSpatialHessianFunction)(inputPoint, jsh, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialHessian()


/**
 * ****************** GetJacobianOfSpatialHessian ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetJacobianOfSpatialHessian(
  const InputPointType &         inputPoint,
  SpatialHessianType &           sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  /** Call the selected GetJacobianOfSpatialHessian. */
  return (this->*m_SelectedGetJacobianOfSpatialHessianFunction2)(inputPoint, sh, jsh, nonZeroJacobianIndices);

} // end GetJacobianOfSpatialHessian()


} // end namespace itk

#endif // end #ifndef itkAdvancedCombinationTransform_hxx
