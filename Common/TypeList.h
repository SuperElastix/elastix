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
// Denis P. Shamonin remarks:
// This file was copied from the ITKSimple, located in
// SimpleITK\src\Code\Common\include\Ancillary\TypeList.h
// Also missing original Loki classes was restored such as:
// (Erase, Replace, NoDuplicates, Reverse).
// Additionally classes VisitDimension and DualVisitDimension was added.
/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
// This file is based off of the work done in the Loki library but is
// substantially modified. It's a good book buy it.
//
////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2001 by Andrei Alexandrescu
// This code accompanies the book:
// Alexandrescu, Andrei. "Modern C++ Design: Generic Programming and Design
//     Patterns Applied". Copyright (c) 2001. Addison-Wesley.
// Permission to use, copy, modify, distribute and sell this software for any
//     purpose is hereby granted without fee, provided that the above copyright
//     notice appear in all copies and that both that copyright notice and this
//     permission notice appear in supporting documentation.
// The author or Addison-Welsey Longman make no representations about the
//     suitability of this software for any purpose. It is provided "as is"
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////

#ifndef TypeList_h
#define TypeList_h

namespace typelist
{

/** \class  TypeList
 * \brief The building block of typelists of any length
 *
 * A TypeList is a type, not an object. It enables complex
 * compile-time manipulation of a set of types.
 *
 * Defines nested types:
 *     Head (first element, a non-typelist type by convention)
 *     Tail (second element, can be another typelist)
 */
template <typename H, typename T>
struct TypeList
{
  using Head = H;
  using Tail = T;
};

/** \class NullType
 * \brief a empty type to indicate end of list
 */
struct NullType
{};

/**\class  MakeTypeList
 * \brief Generates a TypeList from it's template arguments
 *
 * The arguments are type names.
 * \code
 * MakeTypeList<T1, T2, T3>::Type
 * \endcode
 * returns a typelist that contans the types T1, T2, T3
 *
 * Example:
 * \code
 * typedef typelist::MakeTypeList< int, char, short>::Type MyTypeList;
 * \endcode
 *
 */
template <typename T1 = NullType,
          typename T2 = NullType,
          typename T3 = NullType,
          typename T4 = NullType,
          typename T5 = NullType,
          typename T6 = NullType,
          typename T7 = NullType,
          typename T8 = NullType,
          typename T9 = NullType,
          typename T10 = NullType,
          typename T11 = NullType,
          typename T12 = NullType,
          typename T13 = NullType,
          typename T14 = NullType,
          typename T15 = NullType,
          typename T16 = NullType,
          typename T17 = NullType,
          typename T18 = NullType,
          typename T19 = NullType,
          typename T20 = NullType,
          typename T21 = NullType,
          typename T22 = NullType,
          typename T23 = NullType,
          typename T24 = NullType>
struct MakeTypeList
{
private:
  using TailType = typename MakeTypeList<T2,
                                         T3,
                                         T4,
                                         T5,
                                         T6,
                                         T7,
                                         T8,
                                         T9,
                                         T10,
                                         T11,
                                         T12,
                                         T13,
                                         T14,
                                         T15,
                                         T16,
                                         T17,
                                         T18,
                                         T19,
                                         T20,
                                         T21,
                                         T22,
                                         T23,
                                         T24>::Type;

public:
  using Type = TypeList<T1, TailType>;
};
template <>
struct MakeTypeList<>
{
  using Type = NullType;
};

template <typename TTypeList>
struct Length;
/**\class Length
 * \brief Computes the length of a typelist
 *
 * Example:
 * \code
 * typedef typelist::MakeTypeList<int, char, short>::Type MyTypeList;
 * int len = typelist::Length<MyTypeList>::Type;
 * \endcode
 * returns a compile-time constant containing the length of TTypeList,
 * not counting the end terminator (which by convention is NullType)
 *
 */
template <typename H, typename T>
struct Length<TypeList<H, T>>
{
  enum
  {
    Type = 1 + Length<T>::Type
  };
};

/** \cond TYPELIST_IMPLEMENTATION */
template <>
struct Length<NullType>
{
  enum
  {
    Type = 0
  };
};
/** \endcond */

/**\class TypeAt
 * \brief Finds the type at a given index in a typelist
 *
 * Example:
 * \code
 * typedef typelist::MakeTypeList<int, char, short>::Type MyTypeList;
 * typelist::TypeAt<MyTypeList, 0>::Type intVariable;
 * \endcode
 *
 * returns the type's position 'index' in TTypeList
 * If you pass an out-of-bounds index, the result is a compile-time
 * error
 *
 */
template <class TTypeList, unsigned int index>
struct TypeAt;

template <class Head, class Tail>
struct TypeAt<TypeList<Head, Tail>, 0>
{
  using Type = Head;
};

template <class Head, class Tail, unsigned int i>
struct TypeAt<TypeList<Head, Tail>, i>
{
  using Type = typename TypeAt<Tail, i - 1>::Type;
};

template <unsigned int i>
struct TypeAt<NullType, i>
{
  using Type = NullType;
};

template <class TTypeList1, class TTypeList2>
struct Append;
/**\class Append
 * \brief Appends a type or a typelist to another
 *
 * Example 1:
 * \code
 * typedef typelist::MakeTypeList<int, char>::Type MyTypeList1;
 * typedef typelist::MakeTypeList<short, unsigned short>::Type MyTypeList2;
 * typedef typelist::Append<MyList1, MyList2>::Type  MyCombinedList;
 * \endcode
 *
 * Example 2:
 * \code
 * typedef typelist::MakeTypeList<int, char>::Type MyTypeList;
 * typedef typelist::Append<MyTypeList, short>::Type MyAddedTypeList;
 * \endcode
 *
 *  returns a typelist that is TTypeList1 followed by TTypeList2
 *  terminated by NullType. TTypeList2 may be another typelist or a
 *  single type.
 *
 */
template <class Head, class Tail, class T>
struct Append<TypeList<Head, Tail>, T>
{
  using Type = TypeList<Head, typename Append<Tail, T>::Type>;
};

/** \cond TYPELIST_IMPLEMENTATION */
template <>
struct Append<NullType, NullType>
{
  using Type = NullType;
};
template <class T>
struct Append<NullType, T>
{
  using Type = TypeList<T, NullType>;
};
template <class T>
struct Append<T, NullType>
{
  using Type = TypeList<T, NullType>;
};
template <class Head, class Tail>
struct Append<NullType, TypeList<Head, Tail>>
{
  using Type = TypeList<Head, Tail>;
};
template <class Head, class Tail>
struct Append<TypeList<Head, Tail>, NullType>
{
  using Type = TypeList<Head, Tail>;
};

/**\class Erase
 * \brief
 */
template <class TList, class T>
struct Erase;

template <class T>
struct Erase<NullType, T>
{
  using Type = NullType;
};

template <class T, class Tail>
struct Erase<TypeList<T, Tail>, T>
{
  using Type = Tail;
};

template <class Head, class Tail, class T>
struct Erase<TypeList<Head, Tail>, T>
{
  using Type = TypeList<Head, typename Erase<Tail, T>::Type>;
};

/**\class EraseAll
 * \brief
 */
template <class TList, class T>
struct EraseAll;
template <class T>
struct EraseAll<NullType, T>
{
  using Type = NullType;
};
template <class T, class Tail>
struct EraseAll<TypeList<T, Tail>, T>
{
  using Type = typename EraseAll<Tail, T>::Type;
};
template <class Head, class Tail, class T>
struct EraseAll<TypeList<Head, Tail>, T>
{
  using Type = TypeList<Head, typename EraseAll<Tail, T>::Type>;
};

/**\class NoDuplicates
 * \brief
 */
template <class TList>
struct NoDuplicates;

template <>
struct NoDuplicates<NullType>
{
  using Type = NullType;
};

template <class Head, class Tail>
struct NoDuplicates<TypeList<Head, Tail>>
{
private:
  using L1 = typename NoDuplicates<Tail>::Type;
  using L2 = typename Erase<L1, Head>::Type;

public:
  using Type = TypeList<Head, L2>;
};

/**\class Replace
 * \brief
 */
template <class TList, class T, class U>
struct Replace;

template <class T, class U>
struct Replace<NullType, T, U>
{
  using Type = NullType;
};

template <class T, class Tail, class U>
struct Replace<TypeList<T, Tail>, T, U>
{
  using Type = TypeList<U, Tail>;
};

template <class Head, class Tail, class T, class U>
struct Replace<TypeList<Head, Tail>, T, U>
{
  using Type = TypeList<Head, typename Replace<Tail, T, U>::Type>;
};

/**\class ReplaceAll
 * \brief
 */
template <class TList, class T, class U>
struct ReplaceAll;

template <class T, class U>
struct ReplaceAll<NullType, T, U>
{
  using Type = NullType;
};

template <class T, class Tail, class U>
struct ReplaceAll<TypeList<T, Tail>, T, U>
{
  using Type = TypeList<U, typename ReplaceAll<Tail, T, U>::Type>;
};

template <class Head, class Tail, class T, class U>
struct ReplaceAll<TypeList<Head, Tail>, T, U>
{
  using Type = TypeList<Head, typename ReplaceAll<Tail, T, U>::Type>;
};

/**\class Reverse
 * \brief
 */
template <class TList>
struct Reverse;

template <>
struct Reverse<NullType>
{
  using Type = NullType;
};

template <class Head, class Tail>
struct Reverse<TypeList<Head, Tail>>
{
  using Type = typename Append<typename Reverse<Tail>::Type, Head>::Type;
};

/** \endcond */

/**\class IndexOf
 * \brief Finds the index of a type in a typelist
 *
 * Example:
 * \code
 * typedef typelist::MakeTypeList<int, char>::Type MyTypeList;
 * int index = typelist::IndexOf<MyTypeList, int>::Type;
 * \endcode
 *
 * IndexOf<TTypeList, T>::Type
 * returns the position of T in TList, or NullType if T is not found in TList
 */
template <class TTypeList, class TType>
struct IndexOf;
template <class TType>
struct IndexOf<NullType, TType>
{
  enum
  {
    Type = -1
  };
};
template <class TType, class TTail>
struct IndexOf<TypeList<TType, TTail>, TType>
{
  enum
  {
    Type = 0
  };
};
template <class Head, class TTail, class TType>
struct IndexOf<TypeList<Head, TTail>, TType>
{
private:
  enum
  {
    temp = IndexOf<TTail, TType>::Type
  };

public:
  enum
  {
    Type = (temp == -1 ? -1 : 1 + temp)
  };
};

/**\class HasType
 * \brief Queries the typelist for a type
 *
 * Example:
 * \code
 * typedef typelist::MakeTypeList<int, char>::Type MyTypeList;
 * bool query = typelist::HasType<MyTypeList, short>::Type;
 * \endcode
 *
 * HasType<TList, T>::Type
 * evaluates to true if TList contains T, false otherwise.
 */
template <class TTypeList, class TType>
struct HasType;
template <class TType>
struct HasType<NullType, TType>
{
  enum
  {
    Type = false
  };
};
template <class TType, class TTail>
struct HasType<TypeList<TType, TTail>, TType>
{
  enum
  {
    Type = true
  };
};
template <class Head, class TTail, class TType>
struct HasType<TypeList<Head, TTail>, TType>
{
  enum
  {
    Type = HasType<TTail, TType>::Type
  };
};

/**\class Visit
 * \brief Runs a templated predicate on each type in the list
 *
 * \code
 * struct Predicate
 * {
 *  template<class TType>
 *  void operator()() const
 *     { std::cout << typeid(TType).name() << std::endl; }
 * };
 *
 * typedef typelist::MakeTypeList<int, char>::Type MyTypeList;
 * typelist::Visit<MyTypeList>()( Predicate() );
 *
 * \endcode
 *
 *
 */
template <class TTypeList>
struct Visit
{
  template <class Predicate>
  void
  operator()(Predicate & visitor)
  {
    using Head = typename TTypeList::Head;
    using Tail = typename TTypeList::Tail;
    visitor.template operator()<Head>();
    Visit<Tail>      next;
    next.template    operator()<Predicate>(visitor);
  }


  template <class Predicate>
  void
  operator()(const Predicate & visitor)
  {
    using Head = typename TTypeList::Head;
    using Tail = typename TTypeList::Tail;
    visitor.template operator()<Head>();
    Visit<Tail>      next;
    next.template    operator()<Predicate>(visitor);
  }
};

template <>
struct Visit<NullType>
{
  template <class Predicate>
  void
  operator()(const Predicate &)
  {}
};

/**\class VisitDimension
 * \brief Runs a templated predicate on each type in the list
 *        with dimension provided as template parameter.
 * \code
 * struct Predicate
 * {
 *  template<class TType, unsigned int Dimension >
 *  void operator()() const
 *  { std::cout << typeid(TType).name() << ", " << (unsigned int)(Dimension) << std::endl; }
 * };
 *
 * typedef typelist::MakeTypeList<int, char>::Type MyTypeList;
 * typelist::VisitDimension<MyTypeList, 3>()( Predicate() );
 *
 * \endcode
 */
template <class TTypeList, unsigned int Dimension>
struct VisitDimension
{
  template <class Predicate>
  void
  operator()(Predicate & visitor)
  {
    using Head = typename TTypeList::Head;
    using Tail = typename TTypeList::Tail;
    visitor.template                operator()<Head, Dimension>();
    VisitDimension<Tail, Dimension> next;
    next.template                   operator()<Predicate>(visitor);
  }


  template <class Predicate>
  void
  operator()(const Predicate & visitor)
  {
    using Head = typename TTypeList::Head;
    using Tail = typename TTypeList::Tail;
    visitor.template                operator()<Head, Dimension>();
    VisitDimension<Tail, Dimension> next;
    next.template                   operator()<Predicate>(visitor);
  }
};

template <unsigned int Dimension>
struct VisitDimension<NullType, Dimension>
{
  template <class Predicate>
  void
  operator()(const Predicate &)
  {}
};

/**\class DualVisit
 * \brief Runs a templated predicate on each combination of the types
 * on the two lists
 *
 * \code
 * struct Predicate
 * {
 *   template<class TType1, class TType2>
 *     void operator()() const
 *     { std::cout << typeid(TType1).name() << " " << typeid(TType2).name() << std::endl; }
 * };
 *
 * typedef typelist::MakeTypeList<int, char>::Type MyTypeList;
 * typelist::DualVisit<MyTypeList, MyTypeList>()( Predicate() );
 *
 * \endcode
 *
 */
template <typename TLeftTypeList, typename TRightTypeList>
struct DualVisitImpl;

template <typename TLeftTypeList, typename TRightTypeList>
struct DualVisit
{

  template <typename Visitor>
  void
  operator()(Visitor & visitor) const
  {
    DualVisitImpl<TLeftTypeList, TRightTypeList> impl;
    return impl.template                         operator()<Visitor>(visitor);
  }


  template <typename Visitor>
  void
  operator()(const Visitor & visitor) const
  {
    DualVisitImpl<TLeftTypeList, TRightTypeList> impl;
    return impl.template                         operator()<Visitor>(visitor);
  }
};

/** \cond TYPELIST_IMPLEMENTATION
 *
 * The procedural algorithm for this code is:
 * \code
 *  foreach leftType in TLeftTypList
 *    foreach rightType in TRightTypeList
 *      visit( leftType, rightTYpe )
 * \endcode
 *
 * Where inner loop has been unwound in to a tail recursive templated
 * meta-function visitRHS. The outer loop is recursively implemented in
 * the operator().
 */
template <typename TLeftTypeList, typename TRightTypeList>
struct DualVisitImpl
{
  template <typename Visitor>
  void
  operator()(Visitor & visitor) const
  {
    using LeftTail = typename TLeftTypeList::Tail;

    DualVisitImpl<TLeftTypeList, TRightTypeList> goRight;
    goRight.visitRHS<Visitor>(visitor);

    DualVisitImpl<LeftTail, TRightTypeList> goLeft;
    goLeft.template                         operator()<Visitor>(visitor);
  }


  template <typename Visitor>
  void
  operator()(const Visitor & visitor) const
  {
    using LeftTail = typename TLeftTypeList::Tail;

    DualVisitImpl<TLeftTypeList, TRightTypeList> goRight;
    goRight.visitRHS<Visitor>(visitor);

    DualVisitImpl<LeftTail, TRightTypeList> goLeft;
    goLeft.template                         operator()<Visitor>(visitor);
  }


  template <typename Visitor>
  void
  visitRHS(Visitor & visitor) const
  {
    using LeftHead = typename TLeftTypeList::Head;
    using RightHead = typename TRightTypeList::Head;
    using RightTail = typename TRightTypeList::Tail;

    visitor.template operator()<LeftHead, RightHead>();

    DualVisitImpl<TLeftTypeList, RightTail> goRight;
    goRight.template visitRHS<Visitor>(visitor);
  }


  template <typename Visitor>
  void
  visitRHS(const Visitor & visitor) const
  {
    using LeftHead = typename TLeftTypeList::Head;
    using RightHead = typename TRightTypeList::Head;
    using RightTail = typename TRightTypeList::Tail;

    visitor.template operator()<LeftHead, RightHead>();

    DualVisitImpl<TLeftTypeList, RightTail> goRight;
    goRight.template visitRHS<Visitor>(visitor);
  }
};

template <typename TRightTypeList>
struct DualVisitImpl<typelist::NullType, TRightTypeList>
{
  template <typename Visitor>
  void
  operator()(const Visitor &) const
  {}
};
template <typename TLeftTypeList>
struct DualVisitImpl<TLeftTypeList, typelist::NullType>
{
  template <typename Visitor>
  void
  operator()(const Visitor &) const
  {}

  template <typename Visitor>
  void
  visitRHS(const Visitor &) const
  {}
};

template <>
struct DualVisitImpl<typelist::NullType, typelist::NullType>
{
  template <typename Visitor>
  void
  operator()(const Visitor &) const
  {}
};

/**\class DualVisitDimension
 * \brief Runs a templated predicate on each combination of the types
 * on the two lists with dimension provided as template parameter.
 *
 * \code
 * struct Predicate
 * {
 *   template<class TType1, class TType2, unsigned int Dimension>
 *     void operator()() const
 *     { std::cout << typeid(TType1).name()
            << " " << typeid(TType2).name() " "
            << (unsigned int)(VImageDimension) << std::endl; }
 * };
 *
 * typedef typelist::MakeTypeList<int, char>::Type MyTypeList;
 * typelist::DualVisitDimension<MyTypeList, MyTypeList, 2>()( Predicate() );
 *
 * \endcode
 *
 */
template <typename TLeftTypeList, typename TRightTypeList, unsigned int Dimension>
struct DualVisitDimensionImpl;

template <typename TLeftTypeList, typename TRightTypeList, unsigned int Dimension>
struct DualVisitDimension
{

  template <typename Visitor>
  void
  operator()(Visitor & visitor) const
  {
    DualVisitDimensionImpl<TLeftTypeList, TRightTypeList, Dimension> impl;
    return impl.template                                             operator()<Visitor>(visitor);
  }


  template <typename Visitor>
  void
  operator()(const Visitor & visitor) const
  {
    DualVisitDimensionImpl<TLeftTypeList, TRightTypeList, Dimension> impl;
    return impl.template                                             operator()<Visitor>(visitor);
  }
};

/** \cond TYPELIST_IMPLEMENTATION
 *
 * The procedural algorithm for this code is:
 * \code
 *  foreach leftType in TLeftTypList
 *    foreach rightType in TRightTypeList
 *      visit( leftType, rightTYpe )
 * \endcode
 *
 * Where inner loop has been unwound in to a tail recursive templated
 * meta-function visitRHS. The outer loop is recursively implemented in
 * the operator().
 */
template <typename TLeftTypeList, typename TRightTypeList, unsigned int Dimension>
struct DualVisitDimensionImpl
{
  template <typename Visitor>
  void
  operator()(Visitor & visitor) const
  {
    using LeftTail = typename TLeftTypeList::Tail;

    DualVisitDimensionImpl<TLeftTypeList, TRightTypeList, Dimension> goRight;
    goRight.visitRHS<Visitor>(visitor);

    DualVisitDimensionImpl<LeftTail, TRightTypeList, Dimension> goLeft;
    goLeft.template                                             operator()<Visitor>(visitor);
  }


  template <typename Visitor>
  void
  operator()(const Visitor & visitor) const
  {
    using LeftTail = typename TLeftTypeList::Tail;

    DualVisitDimensionImpl<TLeftTypeList, TRightTypeList, Dimension> goRight;
    goRight.visitRHS<Visitor>(visitor);

    DualVisitDimensionImpl<LeftTail, TRightTypeList, Dimension> goLeft;
    goLeft.template                                             operator()<Visitor>(visitor);
  }


  template <typename Visitor>
  void
  visitRHS(Visitor & visitor) const
  {
    using LeftHead = typename TLeftTypeList::Head;
    using RightHead = typename TRightTypeList::Head;
    using RightTail = typename TRightTypeList::Tail;

    visitor.template operator()<LeftHead, RightHead, Dimension>();

    DualVisitDimensionImpl<TLeftTypeList, RightTail, Dimension> goRight;
    goRight.template visitRHS<Visitor>(visitor);
  }


  template <typename Visitor>
  void
  visitRHS(const Visitor & visitor) const
  {
    using LeftHead = typename TLeftTypeList::Head;
    using RightHead = typename TRightTypeList::Head;
    using RightTail = typename TRightTypeList::Tail;

    visitor.template operator()<LeftHead, RightHead, Dimension>();

    DualVisitDimensionImpl<TLeftTypeList, RightTail, Dimension> goRight;
    goRight.template visitRHS<Visitor>(visitor);
  }
};

template <typename TRightTypeList, unsigned int Dimension>
struct DualVisitDimensionImpl<typelist::NullType, TRightTypeList, Dimension>
{
  template <typename Visitor>
  void
  operator()(const Visitor &) const
  {}
};
template <typename TLeftTypeList, unsigned int Dimension>
struct DualVisitDimensionImpl<TLeftTypeList, typelist::NullType, Dimension>
{
  template <typename Visitor>
  void
  operator()(const Visitor &) const
  {}

  template <typename Visitor>
  void
  visitRHS(const Visitor &) const
  {}
};

template <unsigned int Dimension>
struct DualVisitDimensionImpl<typelist::NullType, typelist::NullType, Dimension>
{
  template <typename Visitor>
  void
  operator()(const Visitor &) const
  {}
};

/**\endcond*/

} // namespace typelist

#endif // TypeList_h
