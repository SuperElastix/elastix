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
  if (m_CurrentTransform.IsNotNull())
  {
    return m_CurrentTransform->GetNumberOfParameters();
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
  if (m_CurrentTransform.IsNotNull())
  {
    return m_CurrentTransform->GetNumberOfNonZeroJacobianIndices();
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
  if (m_CurrentTransform.IsNotNull())
  {
    return m_CurrentTransform->GetParameters();
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
  if (m_CurrentTransform.IsNotNull())
  {
    return m_CurrentTransform->GetFixedParameters();
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
  if (m_CurrentTransform.IsNotNull())
  {
    this->Modified();
    m_CurrentTransform->SetParameters(param);
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
  if (m_CurrentTransform.IsNotNull())
  {
    this->Modified();
    m_CurrentTransform->SetFixedParameters(param);
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
  if (m_CurrentTransform.IsNotNull())
  {
    this->Modified();
    m_CurrentTransform->SetParametersByValue(param);
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
  else if (m_CurrentTransform.IsNull())
  {
    /** No current transform has been set. Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }
  else if (m_InitialTransform.IsNull())
  {
    /** No Initial transform, so call the CurrentTransform's implementation. */
    return m_CurrentTransform->GetInverse(inverse);
  }
  else if (m_UseAddition)
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
    //     bool T0invertable = m_InitialTransform->GetInverse( inverseT0 );
    //
    //     if ( T0invertable )
    //     {
    //       /** Try to create the inverse of the current transform. */
    //       CurrentTransformPointer inverseT1 = CurrentTransformType::New();
    //       bool T1invertable = m_CurrentTransform->GetInverse( inverseT1 );
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
  if (m_CurrentTransform.IsNull())
  {
    /** No current transform has been set. Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }
  else if (m_InitialTransform.IsNull())
  {
    /** No Initial transform, so call the CurrentTransform's implementation. */
    return m_CurrentTransform->GetHasNonZeroSpatialHessian();
  }
  else
  {
    bool dummy = m_InitialTransform->GetHasNonZeroSpatialHessian() || m_CurrentTransform->GetHasNonZeroSpatialHessian();
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
  if (m_CurrentTransform.IsNull())
  {
    /** No current transform has been set. Throw an exception. */
    itkExceptionMacro(<< NoCurrentTransformSet);
  }
  else if (m_InitialTransform.IsNull())
  {
    /** No Initial transform, so call the CurrentTransform's implementation. */
    return m_CurrentTransform->GetHasNonZeroJacobianOfSpatialHessian();
  }
  else
  {
    bool dummy = m_InitialTransform->GetHasNonZeroJacobianOfSpatialHessian() ||
                 m_CurrentTransform->GetHasNonZeroJacobianOfSpatialHessian();
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
  if (m_InitialTransform != _arg)
  {
    m_InitialTransform = _arg;
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
  if (m_CurrentTransform != _arg)
  {
    m_CurrentTransform = _arg;
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
  if (m_UseAddition != _arg)
  {
    m_UseAddition = _arg;
    m_UseComposition = !_arg;
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
  if (m_UseComposition != _arg)
  {
    m_UseComposition = _arg;
    m_UseAddition = !_arg;
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
  m_SelectedMethod = m_CurrentTransform.IsNull()
                       ? SelectedMethod::NoCurrentTransform
                       : m_InitialTransform.IsNull()
                           ? SelectedMethod::NoInitialTransform
                           : m_UseAddition ? SelectedMethod::UseAddition : SelectedMethod::UseComposition;

} // end UpdateCombinationMethod()


/**
 * ****************** TransformPoint ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
auto
AdvancedCombinationTransform<TScalarType, NDimensions>::TransformPoint(const InputPointType & point) const
  -> OutputPointType
{
  switch (m_SelectedMethod)
  {
    case SelectedMethod::NoInitialTransform:
    {
      return m_CurrentTransform->TransformPoint(point);
    }
    case SelectedMethod::UseComposition:
    {
      return m_CurrentTransform->TransformPoint(m_InitialTransform->TransformPoint(point));
    }
    case SelectedMethod::UseAddition:
    {
      return m_CurrentTransform->TransformPoint(point) + (m_InitialTransform->TransformPoint(point) - point);
    }
    case SelectedMethod::NoCurrentTransform:
    default:
    {
      itkExceptionMacro(<< NoCurrentTransformSet);
    }
  }

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
  switch (m_SelectedMethod)
  {
    case SelectedMethod::NoInitialTransform:
    case SelectedMethod::UseAddition:
    {
      return m_CurrentTransform->GetJacobian(inputPoint, j, nonZeroJacobianIndices);
    }
    case SelectedMethod::UseComposition:
    {
      return m_CurrentTransform->GetJacobian(m_InitialTransform->TransformPoint(inputPoint), j, nonZeroJacobianIndices);
    }
    case SelectedMethod::NoCurrentTransform:
    default:
    {
      itkExceptionMacro(<< NoCurrentTransformSet);
    }
  }

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
  switch (m_SelectedMethod)
  {
    case SelectedMethod::NoInitialTransform:
    case SelectedMethod::UseAddition:
    {
      return m_CurrentTransform->EvaluateJacobianWithImageGradientProduct(
        inputPoint, movingImageGradient, imageJacobian, nonZeroJacobianIndices);
    }
    case SelectedMethod::UseComposition:
    {
      return m_CurrentTransform->EvaluateJacobianWithImageGradientProduct(
        m_InitialTransform->TransformPoint(inputPoint), movingImageGradient, imageJacobian, nonZeroJacobianIndices);
    }
    case SelectedMethod::NoCurrentTransform:
    default:
    {
      itkExceptionMacro(<< NoCurrentTransformSet);
    }
  }

} // end EvaluateJacobianWithImageGradientProduct()


/**
 * ****************** GetSpatialJacobian ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialJacobian(const InputPointType & inputPoint,
                                                                           SpatialJacobianType &  sj) const
{
  switch (m_SelectedMethod)
  {
    case SelectedMethod::NoInitialTransform:
    {
      return m_CurrentTransform->GetSpatialJacobian(inputPoint, sj);
    }
    case SelectedMethod::UseComposition:
    {
      SpatialJacobianType sj0, sj1;
      m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
      m_CurrentTransform->GetSpatialJacobian(m_InitialTransform->TransformPoint(inputPoint), sj1);
      sj = sj1 * sj0;
      return;
    }
    case SelectedMethod::UseAddition:
    {
      SpatialJacobianType sj0, sj1;
      m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
      m_CurrentTransform->GetSpatialJacobian(inputPoint, sj1);
      sj = sj0 + sj1 - SpatialJacobianType::GetIdentity();
      return;
    }
    case SelectedMethod::NoCurrentTransform:
    default:
    {
      itkExceptionMacro(<< NoCurrentTransformSet);
    }
  }

} // end GetSpatialJacobian()


/**
 * ****************** GetSpatialHessian ****************************
 */

template <typename TScalarType, unsigned int NDimensions>
void
AdvancedCombinationTransform<TScalarType, NDimensions>::GetSpatialHessian(const InputPointType & inputPoint,
                                                                          SpatialHessianType &   sh) const
{
  switch (m_SelectedMethod)
  {
    case SelectedMethod::NoInitialTransform:
    {
      return m_CurrentTransform->GetSpatialHessian(inputPoint, sh);
    }
    case SelectedMethod::UseComposition:
    {
      /** Create intermediary variables for the internal transforms. */
      SpatialJacobianType sj0, sj1;
      SpatialHessianType  sh0, sh1;

      /** Transform the input point. */
      // \todo this has already been computed and it is expensive.
      const InputPointType transformedPoint = m_InitialTransform->TransformPoint(inputPoint);

      /** Compute the (Jacobian of the) spatial Jacobian / Hessian of the
       * internal transforms.
       */
      m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
      m_CurrentTransform->GetSpatialJacobian(transformedPoint, sj1);
      m_InitialTransform->GetSpatialHessian(inputPoint, sh0);
      m_CurrentTransform->GetSpatialHessian(transformedPoint, sh1);

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
      return;
    }
    case SelectedMethod::UseAddition:
    {
      SpatialHessianType sh0, sh1;
      m_InitialTransform->GetSpatialHessian(inputPoint, sh0);
      m_CurrentTransform->GetSpatialHessian(inputPoint, sh1);

      for (unsigned int i = 0; i < SpaceDimension; ++i)
      {
        sh[i] = sh0[i] + sh1[i];
      }

      return;
    }
    case SelectedMethod::NoCurrentTransform:
    default:
    {
      itkExceptionMacro(<< NoCurrentTransformSet);
    }
  }

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
  switch (m_SelectedMethod)
  {
    case SelectedMethod::NoInitialTransform:
    case SelectedMethod::UseAddition:
    {
      return m_CurrentTransform->GetJacobianOfSpatialJacobian(inputPoint, jsj, nonZeroJacobianIndices);
    }
    case SelectedMethod::UseComposition:
    {
      SpatialJacobianType           sj0;
      JacobianOfSpatialJacobianType jsj1;
      m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
      m_CurrentTransform->GetJacobianOfSpatialJacobian(
        m_InitialTransform->TransformPoint(inputPoint), jsj1, nonZeroJacobianIndices);

      jsj.resize(nonZeroJacobianIndices.size());
      for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
      {
        jsj[mu] = jsj1[mu] * sj0;
      }
      return;
    }
    case SelectedMethod::NoCurrentTransform:
    default:
    {
      itkExceptionMacro(<< NoCurrentTransformSet);
    }
  }

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
  switch (m_SelectedMethod)
  {
    case SelectedMethod::NoInitialTransform:
    case SelectedMethod::UseAddition:
    {
      return m_CurrentTransform->GetJacobianOfSpatialJacobian(inputPoint, sj, jsj, nonZeroJacobianIndices);
    }
    case SelectedMethod::UseComposition:
    {
      SpatialJacobianType           sj0, sj1;
      JacobianOfSpatialJacobianType jsj1;
      m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
      m_CurrentTransform->GetJacobianOfSpatialJacobian(
        m_InitialTransform->TransformPoint(inputPoint), sj1, jsj1, nonZeroJacobianIndices);

      sj = sj1 * sj0;
      jsj.resize(nonZeroJacobianIndices.size());
      for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
      {
        jsj[mu] = jsj1[mu] * sj0;
      }
      return;
    }
    case SelectedMethod::NoCurrentTransform:
    default:
    {
      itkExceptionMacro(<< NoCurrentTransformSet);
    }
  }

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
  switch (m_SelectedMethod)
  {
    case SelectedMethod::NoInitialTransform:
    case SelectedMethod::UseAddition:
    {
      return m_CurrentTransform->GetJacobianOfSpatialHessian(inputPoint, jsh, nonZeroJacobianIndices);
    }
    case SelectedMethod::UseComposition:
    {
      /** Create intermediary variables for the internal transforms. */
      SpatialJacobianType           sj0;
      SpatialHessianType            sh0;
      JacobianOfSpatialJacobianType jsj1;
      JacobianOfSpatialHessianType  jsh1;

      /** Transform the input point. */
      // \todo: this has already been computed and it is expensive.
      const InputPointType transformedPoint = m_InitialTransform->TransformPoint(inputPoint);

      /** Compute the (Jacobian of the) spatial Jacobian / Hessian of the
       * internal transforms. */
      m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
      m_InitialTransform->GetSpatialHessian(inputPoint, sh0);

      /** Assume/demand that GetJacobianOfSpatialJacobian returns
       * the same nonZeroJacobianIndices as the GetJacobianOfSpatialHessian. */
      m_CurrentTransform->GetJacobianOfSpatialJacobian(transformedPoint, jsj1, nonZeroJacobianIndices);
      m_CurrentTransform->GetJacobianOfSpatialHessian(transformedPoint, jsh1, nonZeroJacobianIndices);

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

      if (m_InitialTransform->GetHasNonZeroSpatialHessian())
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
      return;
    }
    case SelectedMethod::NoCurrentTransform:
    default:
    {
      itkExceptionMacro(<< NoCurrentTransformSet);
    }
  }

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
  switch (m_SelectedMethod)
  {
    case SelectedMethod::NoInitialTransform:
    case SelectedMethod::UseAddition:
    {
      return m_CurrentTransform->GetJacobianOfSpatialHessian(inputPoint, sh, jsh, nonZeroJacobianIndices);
    }
    case SelectedMethod::UseComposition:
    {
      /** Create intermediary variables for the internal transforms. */
      SpatialJacobianType           sj0, sj1;
      SpatialHessianType            sh0, sh1;
      JacobianOfSpatialJacobianType jsj1;
      JacobianOfSpatialHessianType  jsh1;

      /** Transform the input point. */
      // \todo this has already been computed and it is expensive.
      const InputPointType transformedPoint = m_InitialTransform->TransformPoint(inputPoint);

      /** Compute the (Jacobian of the) spatial Jacobian / Hessian of the
       * internal transforms.
       */
      m_InitialTransform->GetSpatialJacobian(inputPoint, sj0);
      m_InitialTransform->GetSpatialHessian(inputPoint, sh0);

      /** Assume/demand that GetJacobianOfSpatialJacobian returns the same
       * nonZeroJacobianIndices as the GetJacobianOfSpatialHessian.
       */
      m_CurrentTransform->GetJacobianOfSpatialJacobian(transformedPoint, sj1, jsj1, nonZeroJacobianIndices);
      m_CurrentTransform->GetJacobianOfSpatialHessian(transformedPoint, sh1, jsh1, nonZeroJacobianIndices);

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

      if (m_InitialTransform->GetHasNonZeroSpatialHessian())
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

      if (m_InitialTransform->GetHasNonZeroSpatialHessian())
      {
        for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
        {
          for (unsigned int p = 0; p < SpaceDimension; ++p)
          {
            sh[dim] += (sh0[p] * sj1(dim, p));
          }
        }
      }
      return;
    }
    case SelectedMethod::NoCurrentTransform:
    default:
    {
      itkExceptionMacro(<< NoCurrentTransformSet);
    }
  }

} // end GetJacobianOfSpatialHessian()


} // end namespace itk

#endif // end #ifndef itkAdvancedCombinationTransform_hxx
