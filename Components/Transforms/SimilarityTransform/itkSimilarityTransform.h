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
class SimilarityGroup
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
class SimilarityGroup<2>
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
class SimilarityGroup<3>
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
class SimilarityTransform : public SimilarityGroupTemplate<TScalarType, Dimension>
{
public:
  /** Standard ITK-stuff. */
  typedef SimilarityTransform                             Self;
  typedef SimilarityGroupTemplate<TScalarType, Dimension> Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SimilarityTransform, SimilarityGroupTemplate);

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Dimension);

  /** Typedefs inherited from the superclass. */

  /** These are both in Similarity2D and Similarity3D. */
  typedef typename Superclass::ScalarType                ScalarType;
  typedef typename Superclass::ParametersType            ParametersType;
  typedef typename Superclass::NumberOfParametersType    NumberOfParametersType;
  typedef typename Superclass::JacobianType              JacobianType;
  typedef typename Superclass::OffsetType                OffsetType;
  typedef typename Superclass::InputPointType            InputPointType;
  typedef typename Superclass::OutputPointType           OutputPointType;
  typedef typename Superclass::InputVectorType           InputVectorType;
  typedef typename Superclass::OutputVectorType          OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType       OutputVnlVectorType;

  typedef typename Superclass ::NonZeroJacobianIndicesType    NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType            SpatialJacobianType;
  typedef typename Superclass ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType             SpatialHessianType;
  typedef typename Superclass ::JacobianOfSpatialHessianType  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType             InternalMatrixType;

protected:
  SimilarityTransform() = default;
  ~SimilarityTransform() override = default;

private:
  SimilarityTransform(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#endif // end #ifndef itkSimilarityTransform_h
