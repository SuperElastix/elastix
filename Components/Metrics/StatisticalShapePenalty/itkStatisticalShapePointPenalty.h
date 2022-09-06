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
#ifndef itkStatisticalShapePointPenalty_h
#define itkStatisticalShapePointPenalty_h

#include "itkSingleValuedPointSetToPointSetMetric.h"

#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkImage.h"
#include "itkArray.h"
#include <itkVariableSizeMatrix.h>

#include <vnl/vnl_matrix.h>
#include <vnl/vnl_math.h>
#include <vnl/vnl_vector.h>
#include <vnl/algo/vnl_real_eigensystem.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
//#include <vnl/algo/vnl_svd.h>
#include <vnl/algo/vnl_svd_economy.h>

#include <string>

namespace itk
{
/** \class StatisticalShapePointPenalty
 * \brief Computes the Mahalanobis distance between the transformed shape and a mean shape.
 *  A model mean and covariance are required.
 *
 * \author F.F. Berendsen, Image Sciences Institute, UMC Utrecht, The Netherlands
 * \note This work was funded by the projects Care4Me and Mediate.
 * \note If you use the StatisticalShapePenalty anywhere we would appreciate if you cite the following article:\n
 * F.F. Berendsen et al., Free-form image registration regularized by a statistical shape model:
 * application to organ segmentation in cervical MR, Comput. Vis. Image Understand. (2013),
 * http://dx.doi.org/10.1016/j.cviu.2012.12.006
 *
 * \ingroup RegistrationMetrics
 */

template <class TFixedPointSet, class TMovingPointSet>
class ITK_TEMPLATE_EXPORT StatisticalShapePointPenalty
  : public SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(StatisticalShapePointPenalty);

  /** Standard class typedefs. */
  using Self = StatisticalShapePointPenalty;
  using Superclass = SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(StatisticalShapePointPenalty, SingleValuedPointSetToPointSetMetric);

  /** Types transferred from the base class */
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::TransformParametersType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::NonZeroJacobianIndicesType;

  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::DerivativeValueType;
  using typename Superclass::FixedPointSetType;
  using typename Superclass::MovingPointSetType;
  using typename Superclass::FixedPointSetConstPointer;
  using typename Superclass::MovingPointSetConstPointer;

  using typename Superclass::PointIterator;
  using typename Superclass::PointDataIterator;

  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;

  using CoordRepType = typename OutputPointType::CoordRepType;
  using VnlVectorType = vnl_vector<CoordRepType>;
  using VnlMatrixType = vnl_matrix<CoordRepType>;
  // typedef itk::Array<VnlVectorType *> ProposalDerivativeType;
  using ProposalDerivativeType = typename std::vector<VnlVectorType *>;
  // typedef typename vnl_vector<VnlVectorType *> ProposalDerivativeType; //Cannot be linked
  using PCACovarianceType = vnl_svd_economy<CoordRepType>;

  /** Initialization. */
  void
  Initialize() override;

  /**  Get the value for single valued optimizers. */
  MeasureType
  GetValue(const TransformParametersType & parameters) const override;

  /** Get the derivatives of the match measure. */
  void
  GetDerivative(const TransformParametersType & parameters, DerivativeType & Derivative) const override;

  /**  Get value and derivatives for multiple valued optimizers. */
  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                Derivative) const override;

  /** Set/Get the shrinkageIntensity parameter. */
  itkSetClampMacro(ShrinkageIntensity, MeasureType, 0.0, 1.0);
  itkGetConstMacro(ShrinkageIntensity, MeasureType);

  itkSetMacro(ShrinkageIntensityNeedsUpdate, bool);
  itkBooleanMacro(ShrinkageIntensityNeedsUpdate);

  /** Set/Get the BaseVariance parameter. */
  itkSetClampMacro(BaseVariance, MeasureType, -1.0, NumericTraits<MeasureType>::max());
  itkGetConstMacro(BaseVariance, MeasureType);

  itkSetMacro(BaseVarianceNeedsUpdate, bool);
  itkBooleanMacro(BaseVarianceNeedsUpdate);

  itkSetClampMacro(CentroidXVariance, MeasureType, -1.0, NumericTraits<MeasureType>::max());
  itkGetConstMacro(CentroidXVariance, MeasureType);

  itkSetClampMacro(CentroidYVariance, MeasureType, -1.0, NumericTraits<MeasureType>::max());
  itkGetConstMacro(CentroidYVariance, MeasureType);

  itkSetClampMacro(CentroidZVariance, MeasureType, -1.0, NumericTraits<MeasureType>::max());
  itkGetConstMacro(CentroidZVariance, MeasureType);

  itkSetClampMacro(SizeVariance, MeasureType, -1.0, NumericTraits<MeasureType>::max());
  itkGetConstMacro(SizeVariance, MeasureType);

  itkSetMacro(VariancesNeedsUpdate, bool);
  itkBooleanMacro(VariancesNeedsUpdate);

  itkSetClampMacro(CutOffValue, MeasureType, 0.0, NumericTraits<MeasureType>::max());
  itkGetConstMacro(CutOffValue, MeasureType);

  itkSetClampMacro(CutOffSharpness,
                   MeasureType,
                   NumericTraits<MeasureType>::NonpositiveMin(),
                   NumericTraits<MeasureType>::max());
  itkGetConstMacro(CutOffSharpness, MeasureType);

  itkSetMacro(ShapeModelCalculation, int);
  itkGetConstReferenceMacro(ShapeModelCalculation, int);

  itkSetMacro(NormalizedShapeModel, bool);
  itkGetConstReferenceMacro(NormalizedShapeModel, bool);
  itkBooleanMacro(NormalizedShapeModel);

  itkSetConstObjectMacro(EigenVectors, vnl_matrix<double>);
  itkSetConstObjectMacro(EigenValues, vnl_vector<double>);
  itkSetConstObjectMacro(MeanVector, vnl_vector<double>);

  itkSetConstObjectMacro(CovarianceMatrix, vnl_matrix<double>);

protected:
  StatisticalShapePointPenalty();
  ~StatisticalShapePointPenalty() override;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  void
  FillProposalVector(const OutputPointType & fixedPoint, const unsigned int vertexindex) const;

  void
  FillProposalDerivative(const OutputPointType & fixedPoint, const unsigned int vertexindex) const;

  void
  UpdateCentroidAndAlignProposalVector(const unsigned int shapeLength) const;

  void
  UpdateCentroidAndAlignProposalDerivative(const unsigned int shapeLength) const;

  void
  UpdateL2(const unsigned int shapeLength) const;

  void
  NormalizeProposalVector(const unsigned int shapeLength) const;

  void
  UpdateL2AndNormalizeProposalDerivative(const unsigned int shapeLength) const;

  void
  CalculateValue(MeasureType &   value,
                 VnlVectorType & differenceVector,
                 VnlVectorType & centerrotated,
                 VnlVectorType & eigrot) const;

  void
  CalculateDerivative(DerivativeType &      derivative,
                      const MeasureType &   value,
                      const VnlVectorType & differenceVector,
                      const VnlVectorType & centerrotated,
                      const VnlVectorType & eigrot,
                      const unsigned int    shapeLength) const;

  void
  CalculateCutOffValue(MeasureType & value) const;

  void
  CalculateCutOffDerivative(typename DerivativeType::element_type & derivativeElement, const MeasureType & value) const;

  const VnlVectorType * m_MeanVector;
  const VnlMatrixType * m_CovarianceMatrix;
  const VnlMatrixType * m_EigenVectors;
  const VnlVectorType * m_EigenValues;

  VnlMatrixType * m_InverseCovarianceMatrix;

  double m_CentroidXVariance;
  double m_CentroidXStd;
  double m_CentroidYVariance;
  double m_CentroidYStd;
  double m_CentroidZVariance;
  double m_CentroidZStd;
  double m_SizeVariance;
  double m_SizeStd;

  bool m_ShrinkageIntensityNeedsUpdate;
  bool m_BaseVarianceNeedsUpdate;
  bool m_VariancesNeedsUpdate;

  VnlVectorType * m_EigenValuesRegularized;

  mutable ProposalDerivativeType * m_ProposalDerivative;
  unsigned int                     m_ProposalLength;
  bool                             m_NormalizedShapeModel;
  int                              m_ShapeModelCalculation;
  double                           m_ShrinkageIntensity;
  double                           m_BaseVariance;
  double                           m_BaseStd;
  mutable VnlVectorType            m_ProposalVector;
  mutable VnlVectorType            m_MeanValues;

  double m_CutOffValue;
  double m_CutOffSharpness;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkStatisticalShapePointPenalty.hxx"
#endif

#endif
