#ifndef STANDARD_TEMPLATES_CPP
#define STANDARD_TEMPLATES_CPP
/* This file provides a set of standard templates :
 *   IF< condition, then, else> :: RET
 *   AND< A, B >:: RET
 *   OR< A, B>::RET
 *   XOR< A, B> ::RET
 *
 *   empty_type 
 *   which is used in ISEMPTY< type >::RET
 * 
 *   IS_vec< type >::TF
 * 
 * Created by Dirk Poot, Erasmus MC
 */

// IF< condition, A,  B>::RET 
template <bool condition, class Then, class Else> struct IF {
    typedef Then RET;
};
template <class Then, class Else> struct IF<false, Then, Else> {
    typedef Else RET;
};

// AND< A, B>::RET
template <bool conditionA, bool conditionB> struct AND {
	enum {RET = false};
};
template <> struct AND< true, true > {
	enum {RET = true};
};

// OR< A, B>::RET
template <bool conditionA, bool conditionB> struct OR {
	enum {RET = true};
};
template <> struct OR< false, false > {
	enum {RET = false};
};

// XOR< A, B>::RET
template <bool conditionA, bool conditionB> struct XOR {
	enum {RET = true};
};
template <> struct XOR< false, false > {
	enum {RET = false};
};
template <> struct XOR< true, true > {
	enum {RET = false};
};
// EQUAL< A, B >::RET
template <int A, int B> struct EQUAL_INT {
	enum {RET = false};
};
template <int A> struct EQUAL_INT<A,A> {
	enum {RET = true};
};

// WARN< type >::RET
template <typename A> struct WARN {
    static const unsigned Value = -1.0;
    typedef A RET;
};

// Make a dummy/signaling type that should not be actually used. Allow casting of any type to BAD_TYPE.
struct BAD_TYPE { 
	template < typename anyType> BAD_TYPE( anyType dummy) {
		mexErrMsgTxt("Don't create any objects of 'BAD_TYPE'.");
	};
}; 

class empty_type { ; };
template< typename T > struct ISEMPTY {
	enum {RET=false};};
template<> class ISEMPTY<empty_type> {
	enum {RET=true};};

template< typename T> struct STDTYPES {
	typedef typename T::value_type value_type;
};
template <> struct STDTYPES<double> {
	typedef double value_type;
};


#endif