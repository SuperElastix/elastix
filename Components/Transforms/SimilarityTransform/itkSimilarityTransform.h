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

#ifndef itkSimilarityTransform_h
#define itkSimilarityTransform_h

#include "itkAdvancedSimilarity2DTransform.h"
#include "itkAdvancedSimilarity3DTransform.h"
#include "itkAdvancedMatrixOffsetTransformBase.h"

namespace itk
{

/**
 * \class SimilarityGroup
 * \brief This class only contains an alias template.
 *
 */

template <unsigned int Dimension>
class ITK_TEMPLATE_EXPORT SimilarityGroup
{
public:
  template <class TScalarType>
  using TransformAlias = AdvancedMatrixOffsetTransformBase<TScalarType, Dimension, Dimension>;
};

/**
 * \class SimilarityGroup<2>
 * \brief This class only contains an alias template for the 2D case.
 *
 */

template <>
class ITK_TEMPLATE_EXPORT SimilarityGroup<2>
{
public:
  template <class TScalarType>
  using TransformAlias = AdvancedSimilarity2DTransform<TScalarType>;
};

/**
 * \class SimilarityGroup<3>
 * \brief This class only contains an alias template for the 3D case.
 *
 */

template <>
class ITK_TEMPLATE_EXPORT SimilarityGroup<3>
{
public:
  template <class TScalarType>
  using TransformAlias = AdvancedSimilarity3DTransform<TScalarType>;
};


/**
 * This alias templates the SimilarityGroup over its dimension.
 */
template <class TScalarType, unsigned int Dimension>
using SimilarityGroupTemplate = typename SimilarityGroup<Dimension>::template TransformAlias<TScalarType>;


/**
 * \class SimilarityTransform
 * \brief This class combines the Similarity2DTransform with the Similarity3DTransform.
 *
 * This transform is a rigid body transformation, with a uniform scaling.
 *
 * \ingroup Transforms
 */

template <class TScalarType, unsigned int Dimension>
class ITK_TEMPLATE_EXPORT SimilarityTransform : public SimilarityGroupTemplate<TScalarType, Dimension>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SimilarityTransform);

  /** Standard ITK-stuff. */
  using Self = SimilarityTransform;
  using Superclass = SimilarityGroupTemplate<TScalarType, Dimension>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SimilarityTransform, SimilarityGroupTemplate);

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Dimension);

  /** Typedefs inherited from the superclass. */

  /** These are both in Similarity2D and Similarity3D. */
  using typename Superclass::ScalarType;
  using typename Superclass::ParametersType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::JacobianType;
  using typename Superclass::OffsetType;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

protected:
  SimilarityTransform() = default;
  ~SimilarityTransform() override = default;
};

} // end namespace itk

#endif // end #ifndef itkSimilarityTransform_h
