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
#ifndef itkANNFixedRadiusTreeSearch_h
#define itkANNFixedRadiusTreeSearch_h

#include "itkBinaryANNTreeSearchBase.h"

namespace itk
{

/**
 * \class ANNFixedRadiusTreeSearch
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template <class TListSample>
class ITK_TEMPLATE_EXPORT ANNFixedRadiusTreeSearch : public BinaryANNTreeSearchBase<TListSample>
{
public:
  /** Standard itk. */
  typedef ANNFixedRadiusTreeSearch             Self;
  typedef BinaryANNTreeSearchBase<TListSample> Superclass;
  typedef SmartPointer<Self>                   Pointer;
  typedef SmartPointer<const Self>             ConstPointer;

  /** New method for creating an object using a factory. */
  itkNewMacro(Self);

  /** ITK type info. */
  itkTypeMacro(ANNFixedRadiusTreeSearch, BinaryANNTreeSearchBase);

  /** Typedef's from Superclass. */
  typedef typename Superclass::ListSampleType        ListSampleType;
  typedef typename Superclass::BinaryTreeType        BinaryTreeType;
  typedef typename Superclass::MeasurementVectorType MeasurementVectorType;
  typedef typename Superclass::IndexArrayType        IndexArrayType;
  typedef typename Superclass::DistanceArrayType     DistanceArrayType;

  typedef typename Superclass::ANNPointType         ANNPointType;         // double *
  typedef typename Superclass::ANNIndexType         ANNIndexType;         // int
  typedef typename Superclass::ANNIndexArrayType    ANNIndexArrayType;    // int *
  typedef typename Superclass::ANNDistanceType      ANNDistanceType;      // double
  typedef typename Superclass::ANNDistanceArrayType ANNDistanceArrayType; // double *

  typedef typename Superclass::BinaryANNTreeType BinaryANNTreeType;

  /** Set and get the error bound eps. */
  itkSetClampMacro(ErrorBound, double, 0.0, 1e14);
  itkGetConstMacro(ErrorBound, double);

  /** Set and get the squared radius search bound. */
  itkSetMacro(SquaredRadius, double);
  itkGetConstMacro(SquaredRadius, double);

  /** Search the nearest neighbours of a query point qp. */
  void
  Search(const MeasurementVectorType & qp, IndexArrayType & ind, DistanceArrayType & dists) override;

  /** Search the nearest neighbours of a query point qp. */
  virtual void
  Search(const MeasurementVectorType & qp, IndexArrayType & ind, DistanceArrayType & dists, double sqRad);

protected:
  ANNFixedRadiusTreeSearch();
  ~ANNFixedRadiusTreeSearch() override;

  /** Member variables. */
  double m_ErrorBound;
  double m_SquaredRadius;

private:
  ANNFixedRadiusTreeSearch(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkANNFixedRadiusTreeSearch.hxx"
#endif

#endif // end #ifndef itkANNFixedRadiusTreeSearch_h
